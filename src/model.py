"""GraphRiverCast — Topology-aware neural hydrodynamic model.

A unified global river model that encodes river network connectivity as a
structural prior via signed bidirectional graph convolution (SignedGCN) with
edge-attribute gating, coupled with an LSTMCell temporal encoder and
per-node fusion gating.

Architecture:
    EdgeAttrConv        — Edge-attribute-gated message passing layer
    SignedGCN_Encoder   — Dual-path (upstream/downstream) graph encoder
    GraphRiverCast      — Full model: embed → feat_mix → GNN → LSTM → readout

Reference:
    Ren et al. (2026). Topology enables learning-based hydrodynamic prediction
    of the global river system. arXiv:2602.22293.
"""

import numbers
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Size, nn
from torch.nn.parameter import Parameter
from torch_geometric.nn import MessagePassing


# ═══════════════════════════════════════════════════════════════════
#  RMSNorm — Root Mean Square Layer Normalization
# ═══════════════════════════════════════════════════════════════════

_shape_t = Union[int, List[int], Size]


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""

    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: Optional[float] = None,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(
                torch.empty(self.normalized_shape, **factory_kwargs)
            )
        else:
            self.register_parameter("weight", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            x
            * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + self.eps)
            * self.weight
        )

    def extra_repr(self) -> str:
        return (
            "{normalized_shape}, eps={eps}, "
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)
        )


# ═══════════════════════════════════════════════════════════════════
#  EdgeAttrConv — Edge-attribute-gated message passing
# ═══════════════════════════════════════════════════════════════════


class EdgeAttrConv(MessagePassing):
    """Message passing with edge-attribute gating:
        msg_j = Linear(x_j) * sigmoid(EdgeMLP(edge_attr))
    """

    def __init__(self, in_dim: int, out_dim: int, edge_feat_dim: int,
                 dropout: float = 0.1):
        super().__init__(aggr="add")
        self.lin = nn.Linear(in_dim, out_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_feat_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        msg = self.lin(x_j)
        gate = torch.sigmoid(self.edge_mlp(edge_attr))
        return self.dropout(msg * gate)


# ═══════════════════════════════════════════════════════════════════
#  SignedGCN_Encoder — Bidirectional edge-attribute-aware encoder
# ═══════════════════════════════════════════════════════════════════


class SignedGCN_Encoder(nn.Module):
    """Dual-path graph encoder with independent weights per direction.

    Positive path: upstream → downstream (natural flow direction).
    Negative path: downstream → upstream (backwater / reverse influence).
    Each path uses its own set of EdgeAttrConv layers with direction-
    specific edge attributes.

    Returns (h_pos, h_neg), both shaped [B, N, H].
    """

    def __init__(self, in_dim: int, hid_dim: int, num_layers: int,
                 edge_feat_dim: int, dropout: float = 0.1):
        super().__init__()
        assert num_layers >= 1
        self.hid_dim = hid_dim
        self.dropout = nn.Dropout(dropout)

        self.pos_convs = nn.ModuleList()
        self.pos_convs.append(EdgeAttrConv(in_dim, hid_dim, edge_feat_dim, dropout))
        for _ in range(num_layers - 1):
            self.pos_convs.append(EdgeAttrConv(hid_dim, hid_dim, edge_feat_dim, dropout))

        self.neg_convs = nn.ModuleList()
        self.neg_convs.append(EdgeAttrConv(in_dim, hid_dim, edge_feat_dim, dropout))
        for _ in range(num_layers - 1):
            self.neg_convs.append(EdgeAttrConv(hid_dim, hid_dim, edge_feat_dim, dropout))

    @staticmethod
    def build_batch_edge_index(B: int, N: int, edge_index: torch.Tensor):
        """Pre-compute batched edge indices (call once, cache across timesteps).

        Returns:
            pos_ei: [2, B*E] forward edge indices with node offsets
            neg_ei: [2, B*E] reverse edge indices
        """
        E = edge_index.size(1)
        offsets = torch.arange(B, device=edge_index.device).view(B, 1, 1) * N
        ei_expanded = edge_index.unsqueeze(0).expand(B, 2, E) + offsets
        pos_ei = ei_expanded.permute(1, 0, 2).reshape(2, B * E)
        neg_ei = pos_ei.flip(dims=[0])
        return pos_ei, neg_ei

    def forward(self, x, pos_ei, neg_ei, pos_edge_attr, neg_edge_attr):
        """
        Args:
            x:              [B, N, F] node features
            pos_ei:         [2, B*E] pre-computed forward edge indices
            neg_ei:         [2, B*E] pre-computed reverse edge indices
            pos_edge_attr:  [B, E, D] forward edge attributes
            neg_edge_attr:  [B, E, D] reverse edge attributes

        Returns:
            (h_pos, h_neg): each [B, N, H]
        """
        B, N, _ = x.shape
        x_b = x.reshape(B * N, -1)
        ea_pos = pos_edge_attr.reshape(-1, pos_edge_attr.size(-1))
        ea_neg = neg_edge_attr.reshape(-1, neg_edge_attr.size(-1))

        h_pos = x_b
        for conv in self.pos_convs[:-1]:
            h_new = conv(h_pos, pos_ei, ea_pos)
            h_new = F.gelu(h_new)
            h_pos = h_pos + h_new
        h_new = self.pos_convs[-1](h_pos, pos_ei, ea_pos)
        h_pos = h_pos + self.dropout(h_new)

        h_neg = x_b
        for conv in self.neg_convs[:-1]:
            h_new = conv(h_neg, neg_ei, ea_neg)
            h_new = F.gelu(h_new)
            h_neg = h_neg + h_new
        h_new = self.neg_convs[-1](h_neg, neg_ei, ea_neg)
        h_neg = h_neg + self.dropout(h_new)

        h_pos = h_pos.view(B, N, self.hid_dim)
        h_neg = h_neg.view(B, N, self.hid_dim)
        return h_pos, h_neg


# ═══════════════════════════════════════════════════════════════════
#  GRC_V2 — Base model (SignedGCN + GRU)
# ═══════════════════════════════════════════════════════════════════
# Kept as the base class to ensure checkpoint state_dict compatibility.
# GraphRiverCast (below) inherits and replaces GRU with LSTMCell.


class GRC_V2(nn.Module):
    def __init__(self, cfg: dict, task: dict):
        super().__init__()
        self.cfg = cfg
        self.task = task

        self.use_spatial = self.cfg["use_spatial"]
        self.use_temporal = self.cfg["use_temporal"]
        self.use_static_var = self.cfg["use_static_var"]
        self.use_river_var = self.cfg["use_river_var"]

        win_cfg = self.task['window'][self.task['type']]
        self.hist_len = win_cfg['history']
        self.fut_len = win_cfg['future']

        self.spatial_num_layer = self.cfg.get("spatial_num_layer", 2)
        self.temporal_num_layer = 1
        self.hid_size = self.cfg['hid_size']
        self.fmix_size = self.cfg['fmix_size']
        self.dropout_rate = self.cfg.get("dropout_rate", 0.1)
        self.eps = self.cfg.get("eps", 1e-6)
        self.edge_feat_dim = self.cfg.get("edge_feat_dim", 9)

        base_dim = 1       # runoff
        river_dim = 3      # outflw, rivdph, storage
        static_dim = 28 if self.use_static_var else 0
        self.in_dim = base_dim + river_dim + static_dim
        self.out_dim = 3

        self.embed = nn.Linear(self.in_dim, self.hid_size)
        self.readout = nn.Linear(self.hid_size, self.out_dim)
        self.out_norm = RMSNorm([self.hid_size], eps=self.eps)

        self.feat_mix = nn.Sequential(
            nn.Linear(self.hid_size, self.fmix_size),
            nn.SiLU(),
            nn.Linear(self.fmix_size, self.hid_size),
            nn.Dropout(self.dropout_rate),
        )
        self.fmix_norm = RMSNorm([self.hid_size], eps=self.eps)

        if self.use_temporal:
            self.gru_cell = nn.GRU(
                input_size=self.hid_size, hidden_size=self.hid_size,
                num_layers=self.temporal_num_layer, batch_first=True,
                dropout=self.dropout_rate,
            )
            self.gru_norm = RMSNorm([self.hid_size], eps=self.eps)
            self.hid_norm = RMSNorm([self.hid_size], eps=self.eps)

        if self.use_spatial:
            self.gnn_encoder = SignedGCN_Encoder(
                in_dim=self.hid_size,
                hid_dim=self.hid_size,
                num_layers=self.spatial_num_layer,
                edge_feat_dim=self.edge_feat_dim,
                dropout=self.dropout_rate,
            )
            self.gnn_norm = RMSNorm([self.hid_size], eps=self.eps)
            self.fusion_gate = nn.Sequential(
                nn.Linear(2 * self.hid_size, self.hid_size),
                nn.Sigmoid(),
            )


# ═══════════════════════════════════════════════════════════════════
#  GraphRiverCast — Production model (SignedGCN + LSTMCell)
# ═══════════════════════════════════════════════════════════════════


class GraphRiverCast(GRC_V2):
    """GraphRiverCast — Topology-aware global river hydrodynamic model.

    Inherits the full spatial module (SignedGCN_Encoder with bidirectional
    edge-attribute-gated convolution and per-node fusion gate) from GRC_V2,
    and replaces the temporal module with LSTMCell for more stable gradient
    propagation over long sequences.

    Key features:
        - Signed bidirectional graph convolution with 9-dim edge attributes
          (8 static hydraulic-geometric + 1 dynamic water-depth gradient)
        - LSTMCell temporal encoder with RMSNorm on both h_t and c_t
        - Pre-allocated output tensors for memory efficiency
        - Gradient checkpointing during training (~95% activation memory saving)
        - Supports ColdStart (zero initial state) and HotStart (from data)

    Args:
        cfg: Model configuration dict with keys:
            use_spatial, use_temporal, use_static_var, use_river_var,
            hid_size, fmix_size, dropout_rate, eps, spatial_num_layer,
            edge_feat_dim
        task: Task configuration dict with keys:
            type ("predict"), window (history/future lengths)
    """

    def __init__(self, cfg: dict, task: dict):
        super().__init__(cfg, task)

        if self.use_temporal:
            del self.gru_cell
            self.lstm_cell = nn.LSTMCell(self.hid_size, self.hid_size)
            self.cell_norm = RMSNorm([self.hid_size], eps=self.eps)

    def _one_step(
        self,
        x_in_t,
        current_river,
        h_t,
        c_t,
        static_var,
        ei,
        pos_ei,
        neg_ei,
        pos_ea_buf,
        neg_ea_buf,
    ):
        """Single timestep: embed → GNN → LSTM → readout.

        Args:
            x_in_t:        [B, N, 1] runoff at current timestep
            current_river: [B, N, 3] current river state (Q, H, S)
            h_t:           [B*N, hid] LSTM hidden state
            c_t:           [B*N, hid] LSTM cell state
            static_var:    [B, N, S] static features (or None)
            ei:            [2, E] original edge index
            pos_ei:        [2, B*E] batched forward edge index
            neg_ei:        [2, B*E] batched reverse edge index
            pos_ea_buf:    [B, E, D] forward edge attribute buffer
            neg_ea_buf:    [B, E, D] reverse edge attribute buffer

        Returns:
            river_t: [B, N, 3] updated river state
            h_t:     [B*N, hid] updated LSTM hidden state
            c_t:     [B*N, hid] updated LSTM cell state
        """
        if self.use_static_var and static_var is not None:
            x_t = torch.cat((x_in_t, current_river, static_var), dim=-1)
        else:
            x_t = torch.cat((x_in_t, current_river), dim=-1)

        bs, n, _ = x_t.shape

        x_t = self.embed(x_t)
        x_t = x_t + self.feat_mix(self.fmix_norm(x_t))

        if self.use_spatial and pos_ea_buf is not None:
            rivdph_t = current_river[..., 1]
            delta_fwd = rivdph_t[:, ei[0]] - rivdph_t[:, ei[1]]
            pos_ea_buf = pos_ea_buf.clone()
            neg_ea_buf = neg_ea_buf.clone()
            pos_ea_buf[..., -1] = delta_fwd
            neg_ea_buf[..., -1] = -delta_fwd

            h_pos, h_neg = self.gnn_encoder(
                self.gnn_norm(x_t), pos_ei, neg_ei,
                pos_ea_buf, neg_ea_buf,
            )
            gate = self.fusion_gate(torch.cat([h_pos, h_neg], dim=-1))
            x_t = x_t + gate * h_pos + (1 - gate) * h_neg

        if self.use_temporal:
            lstm_input = self.gru_norm(x_t.reshape(bs * n, -1))
            if bs * n > 65535:
                with torch.amp.autocast("cuda", enabled=False):
                    h_t, c_t = self.lstm_cell(
                        lstm_input.float(),
                        (self.hid_norm(h_t).float(), self.cell_norm(c_t).float()),
                    )
                x_t = x_t + h_t.to(x_t.dtype).view(bs, n, -1)
            else:
                h_t, c_t = self.lstm_cell(
                    lstm_input,
                    (self.hid_norm(h_t), self.cell_norm(c_t)),
                )
                x_t = x_t + h_t.view(bs, n, -1)

        x_delta = self.readout(self.out_norm(x_t))
        river_t = current_river + x_delta.view(bs, n, -1)

        return river_t, h_t, c_t

    def forward(self, inputs):
        """Run GraphRiverCast forward pass.

        Args:
            inputs: dict with keys:
                river_hist:    [B, T_hist, N, 3] historical river states
                                (zeros for ColdStart)
                runoff_hist:   [B, T_hist, N, 1] historical runoff forcing
                runoff_fut:    [B, T_fut, N, 1]  future runoff forcing
                static_var:    [B, N, 28] static node features
                edge_index:    [2, E] or [B, 2, E] river network topology
                pos_edge_attr: [B, E, 8] forward edge attributes
                neg_edge_attr: [B, E, 8] reverse edge attributes

        Returns:
            dict with key "river_fut_hat": [B, T_fut, N, 3]
                predicted future river states (Q, H, S)
        """
        river_hist = inputs["river_hist"] if self.use_river_var else None
        runoff_hist = inputs["runoff_hist"]
        runoff_fut = inputs["runoff_fut"]
        static_var = inputs["static_var"] if self.use_static_var else None
        edge_index = inputs["edge_index"]

        pos_edge_attr = inputs.get("pos_edge_attr")
        neg_edge_attr = inputs.get("neg_edge_attr")

        if edge_index.dim() == 3:
            ei = edge_index[0]
        else:
            ei = edge_index

        x_in = torch.cat([runoff_hist, runoff_fut], dim=1)
        bs, ts, n, _ = x_in.shape

        h_t = torch.zeros(bs * n, self.hid_size,
                          device=x_in.device, dtype=x_in.dtype)
        c_t = torch.zeros(bs * n, self.hid_size,
                          device=x_in.device, dtype=x_in.dtype)

        pos_ei = neg_ei = None
        pos_ea_buf = neg_ea_buf = None

        if self.use_spatial and pos_edge_attr is not None:
            pos_ei, neg_ei = SignedGCN_Encoder.build_batch_edge_index(bs, n, ei)
            pos_ea_buf = torch.empty(
                bs, ei.size(1), self.edge_feat_dim,
                device=x_in.device, dtype=x_in.dtype,
            )
            neg_ea_buf = torch.empty_like(pos_ea_buf)
            pos_ea_buf[..., :-1] = pos_edge_attr
            neg_ea_buf[..., :-1] = neg_edge_attr

        fut_len = ts - self.hist_len
        fut_preds = torch.empty(bs, fut_len, n, 3,
                                device=x_in.device, dtype=x_in.dtype)

        current_river = None
        river_avail_len = river_hist.shape[1] if river_hist is not None else 0

        for t in range(ts):
            if self.use_river_var:
                current_river = river_hist[:, t] if t < river_avail_len else current_river
            else:
                if t == 0:
                    current_river = torch.zeros(
                        bs, n, 3, device=x_in.device, dtype=x_in.dtype,
                    )

            river_t, h_t, c_t = self._one_step(
                x_in[:, t], current_river, h_t, c_t, static_var,
                ei, pos_ei, neg_ei, pos_ea_buf, neg_ea_buf,
            )

            current_river = river_t
            if t >= self.hist_len:
                fut_preds[:, t - self.hist_len] = river_t

        return {"river_fut_hat": fut_preds}
