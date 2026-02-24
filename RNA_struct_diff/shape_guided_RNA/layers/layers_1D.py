import math

import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange
from inspect import isfunction



def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d




class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class Rezero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.zeros(1))

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) * self.g




class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, num_steps, rescale_steps=4000):
        super().__init__()
        self.dim = dim
        self.num_steps = float(num_steps)
        self.rescale_steps = float(rescale_steps)

    def forward(self, x):
        x = x / self.num_steps * self.rescale_steps
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(dim, dim_out, kernel_size=3, padding=1),
            nn.GroupNorm(groups, dim_out),
            Mish()
        )
    def forward(self, x):
        return self.block(x)



class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            Mish(),
            nn.Linear(time_emb_dim, dim_out)
        )
        self.block1 = Block(dim, dim_out, groups)
        self.block2 = Block(dim_out, dim_out, groups)
        self.res_conv = nn.Conv1d(dim, dim_out, kernel_size=1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        h = h + self.mlp(time_emb)[:, :, None]   # <— 1D: add along length
        h = self.block2(h)
        return h + self.res_conv(x)


class LinearAttention1D(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, gate_init=0.1):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner = heads * dim_head
        self.to_q = nn.Conv1d(dim, inner, 1, bias=False)
        self.to_k = nn.Conv1d(dim, inner, 1, bias=False)
        self.to_v = nn.Conv1d(dim, inner, 1, bias=False)
        self.to_o = nn.Conv1d(inner, dim, 1)
        self.gate = nn.Parameter(torch.tensor(gate_init))
        self.alpha = nn.Parameter(torch.tensor(gate_init))
    @staticmethod
    def _kernel(x):  # Performer-style positive feature map
        return F.elu(x) + 1.0

    def forward(self, x, guidance=None, pad_mask=None):
        B, C, L = x.shape
        H, Dh = self.heads, self.dim_head
        q = self._kernel(self.to_q(x).view(B, H, Dh, L))
        k = self._kernel(self.to_k(x).view(B, H, Dh, L))
        v =               self.to_v(x).view(B, H, Dh, L)

        # cheap token-wise gate from guidance (degree), optional
        if guidance is not None:
            #include a check for RNAfold
            g = guidance.float()
            if g.min() >= 0:
                deg = g.sum(-1)  # EXACT old behaviour
            else:
                # new behaviour for signed maps: ignore negative mass for degree
                deg = g.clamp(min=0.0).sum(-1)  # (B, L)

            #deg = guidance.float().sum(-1)  # (B, L)
            if pad_mask is not None:
                deg = deg * pad_mask.float()
            k = k * (1.0 + self.gate * deg.view(B, 1, 1, L))
            q = q * (1.0 + self.alpha * deg.view(B, 1, 1, L))
        kv = torch.einsum('b h d l, b h e l -> b h d e', k, v)
        z  = 1.0 / (torch.einsum('b h d l, b h d -> b h l', q, k.sum(-1)) + 1e-6)
        out = torch.einsum('b h d e, b h d l -> b h e l', kv, q) * z[:, :, None, :]
        return self.to_o(out.reshape(B, H*Dh, L))


class TruePairwiseAttention1D(nn.Module):
    """Dense softmax attention with additive LxL bias — used only at bottleneck."""
    def __init__(self, dim, heads=4, dim_head=32, bias_scale_init=1.0):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner = heads * dim_head
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_k = nn.Linear(dim, inner, bias=False)
        self.to_v = nn.Linear(dim, inner, bias=False)
        self.to_o = nn.Linear(inner, dim)
        self.bias_scale = nn.Parameter(torch.tensor(bias_scale_init))

    def forward(self, x, guidance=None, pad_mask=None):
        # x: (B, C, L)  -> do attention in (B, L, C)
        x = x.transpose(1, 2)
        B, L, C = x.shape
        H, Dh = self.heads, self.dim_head

        q = self.to_q(x).view(B, L, H, Dh).transpose(1, 2)  # (B,H,L,Dh)
        k = self.to_k(x).view(B, L, H, Dh).transpose(1, 2)
        v = self.to_v(x).view(B, L, H, Dh).transpose(1, 2)

        scores = torch.einsum('b h i d, b h j d -> b h i j', q, k) / math.sqrt(Dh)

        if guidance is not None:
            scores = scores + self.bias_scale * guidance[:, None, :, :]
        if pad_mask is not None:
            key_mask = pad_mask[:, None, None, :]  # (B, 1, 1, L)
            scores = scores.masked_fill(~key_mask, float('-inf'))

        attn = scores.softmax(dim=-1)
        out  = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        if pad_mask is not None:
            q_mask = pad_mask[:, None, :, None].float()  # (B,1,L,1)
            out = out * q_mask
        out  = out.transpose(1, 2).contiguous().view(B, L, H*Dh)
        out  = self.to_o(out)
        return out.transpose(1, 2)  # back to (B, C, L)


class TruePairwiseAttention1DAdapter(nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.attn = attn  # expects (B, L, C)

    def forward(self, x, guidance=None, pad_mask=None):
        # x: (B, C, L)
        y = self.attn(x.transpose(1, 2), guidance=guidance, pad_mask=pad_mask)  # (B, L, C)
        return y.transpose(1, 2)  # (B, C, L)


class FiLM1D(nn.Module):
    def __init__(self, c):
        super().__init__()
        # input: (B, 1, L) degree feature → outputs (B, 2C, L) -> split to γ, β
        self.proj = nn.Conv1d(1, 2*c, kernel_size=1)
        self.gate = nn.Parameter(torch.tensor(0.1))  # small init

    def forward(self, x, deg_norm):  # x: (B,C,L), deg_norm: (B,L) in [0,1]
        d = deg_norm[:, None, :]                # (B,1,L)
        gb = self.proj(d)                       # (B,2C,L)
        gamma, beta = gb.chunk(2, dim=1)        # (B,C,L) each
        # stabilize: γ ≈ 1 initially
        gamma = 1.0 + 0.1 * torch.tanh(gamma)
        beta  = 0.1 * torch.tanh(beta)
        return x + self.gate * ((gamma - 1.0) * x + beta)
        #return x + self.gate * (gamma * x + beta)




def contactmap_to_partner_idx_1d(cm: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    # cm: (B,L,L) already padded/zeroed outside true length
    # pad_mask: (B,L) True for valid tokens
    j_star = cm.argmax(dim=-1)                               # (B,L)
    has_partner = (cm.sum(dim=-1) > 0.5) & pad_mask.bool()   # (B,L)
    partner_idx = torch.full_like(j_star, -1)                # -1 sentinel
    partner_idx[has_partner] = j_star[has_partner]
    partner_idx[~pad_mask.bool()] = -1
    return partner_idx
'''
def signed_distance_feats_1d(partner_idx: torch.Tensor, pad_mask: torch.Tensor, K: int = 2) -> torch.Tensor:
    """
    Per-token WHERE features; output (B, 2+2K, L):
    [paired_mask, delta, sin(2πkΔ), cos(2πkΔ) for k=1..K]
    with Δ = (j-i)/max(1, L_eff-1); zeros on pads/unpaired.
    """
    B, L = partner_idx.shape
    device = partner_idx.device
    i = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)  # (B,L)
    paired = (partner_idx >= 0) & pad_mask.bool()

    # effective length per batch (non-pad tokens)
    L_eff = pad_mask.sum(dim=1, keepdim=True).clamp_min(1)  # (B,1)

    delta = torch.zeros_like(i, dtype=torch.float32)
    delta[paired] = (partner_idx[paired] - i[paired]).float() / (L_eff.expand_as(i)[paired] - 1).clamp_min(1)

    feats = [paired.float(), delta]
    for k in range(1, K + 1):
        w = 2 * math.pi * k
        feats.append(torch.sin(w * delta))
        feats.append(torch.cos(w * delta))
    # zero out pads
    feats = [f * pad_mask.float() for f in feats]
    return torch.stack(feats, dim=1)  # (B, 2+2K, L)
'''


def signed_distance_feats_1d(
    partner_idx: torch.Tensor,
    pad_mask: torch.Tensor,
    K: int = 2,
    absolute: bool = False,   # <- switch: False = relative-to-length, True = absolute-in-nt
    cap: int = 256,           # used only if absolute=True
) -> torch.Tensor:
    """
    Per-token WHERE features; output (B, 2+2K, L):
      [paired_mask, delta, sin(2πkΔ), cos(2πkΔ) for k=1..K]
    Δ =
      relative=False: (j - i) / max(1, L_eff - 1)     # normalized by effective length
      absolute=True : clamp(j - i, [-cap, cap]) / cap # absolute in nt, scaled and clipped
    Pads/unpaired → zeros.
    """
    B, L = partner_idx.shape
    device = partner_idx.device

    i = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)   # (B,L)
    paired = (partner_idx >= 0) & pad_mask.bool()                   # valid pairs

    # signed offsets
    d = torch.zeros_like(i, dtype=torch.float32)
    d[paired] = (partner_idx[paired] - i[paired]).float()

    if absolute:
        denom = float(max(cap, 1))
        delta = d.clamp(-denom, denom) / denom
    else:
        # effective non-pad length per batch item
        L_eff = pad_mask.sum(dim=1, keepdim=True).clamp_min(1)      # (B,1)
        denom = (L_eff.expand_as(i) - 1).clamp_min(1).float()
        delta = torch.zeros_like(d)
        delta[paired] = d[paired] / denom[paired]

    feats = [paired.float(), delta]
    for k in range(1, K + 1):
        w = 2 * math.pi * k
        feats.append(torch.sin(w * delta))
        feats.append(torch.cos(w * delta))

    m = pad_mask.float()
    feats = [f * m for f in feats]                                  # zero pads
    return torch.stack(feats, dim=1)



class AbsPosCache1D(nn.Module):
    def __init__(self, pos_dim: int = 32):
        super().__init__()
        assert pos_dim % 2 == 0
        self.pos_dim = pos_dim
        self.cache = {}
    def forward(self, L: int, device):
        if L in self.cache: return self.cache[L]
        pos = torch.arange(L, device=device).float()
        div = torch.exp(torch.arange(0, self.pos_dim, 2, device=device).float()
                        * (-math.log(10000.0) / self.pos_dim))
        pe = torch.zeros(self.pos_dim, L, device=device)
        pe[0::2] = torch.sin(pos[None] * div[:, None])
        pe[1::2] = torch.cos(pos[None] * div[:, None])
        pe = pe.unsqueeze(0)  # (1,C,L)
        self.cache[L] = pe
        return pe

class PairRouter1D(nn.Module):
    """
    Padding-safe pair routing for 1D streams with optional distance-aware gating.
    x:           (B, C, L)
    partner_idx: (B, L)  long; -1 = unpaired/pad  (IMPORTANT: do not use 0)
    pad_mask:    (B, L)  bool or {0,1}
    dist_feats:  (B, F, L) from signed_distance_feats_1d (optional)
    """
    def __init__(self, dim: int, dist_feat_dim: int = 0, hidden: int = 0, gate_init: float = 0.0):
        super().__init__()
        h = hidden or dim
        self.mix = nn.Sequential(
            nn.Conv1d(dim * 2, h, 1, bias=True),
            nn.SiLU(),
            nn.Conv1d(h, dim, 1, bias=True),
        )
        self.gate_global = nn.Parameter(torch.tensor(gate_init, dtype=torch.float32))
        self.has_dist = dist_feat_dim > 0
        if self.has_dist:
            self.gate_dist = nn.Sequential(
                nn.Conv1d(dist_feat_dim, dim, 1, bias=True),
                nn.SiLU(),
                nn.Conv1d(dim, 1, 1, bias=True),  # (B,1,L)
            )

    @staticmethod
    def _gather_len(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        # Clamp for safety; masked out later anyway
        idx = idx.clamp(0, L-1)
        return torch.gather(x, 2, idx.unsqueeze(1).expand(B, C, L))

    def forward(self, x: torch.Tensor, partner_idx: torch.Tensor,
                pad_mask: torch.Tensor, dist_feats: torch.Tensor = None) -> torch.Tensor:
        B, C, L = x.shape
        device = x.device
        pad_mask = pad_mask.to(x.dtype)  # (B,L)

        # valid sources and safe index (fallback to self index for unpaired/pad)
        i_idx = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        has_partner = (partner_idx >= 0) & pad_mask.bool()
        safe_idx = torch.where(has_partner, partner_idx, i_idx)            # (B,L)

        # gather partner features and mix
        x_partner = self._gather_len(x, safe_idx)                          # (B,C,L)
        y = self.mix(torch.cat([x, x_partner], dim=1))                     # (B,C,L)

        # residual mask: both endpoints valid (avoid leakage from pads)
        partner_valid = torch.gather(pad_mask, 1, safe_idx)
        both_valid = (pad_mask * partner_valid).unsqueeze(1)               # (B,1,L)

        # gates: global + optional distance-aware per-token
        g = torch.tanh(self.gate_global)                                   # scalar
        if self.has_dist and dist_feats is not None:
            gd = torch.sigmoid(self.gate_dist(dist_feats))                 # (B,1,L)
            g = g * gd                                                     # broadcast per-token

        y = y * both_valid
        out = (x + g * y) * pad_mask.unsqueeze(1)                          # keep pads at zero
        return out





class SegmentationUnet1D(nn.Module):
    def __init__(self, num_classes, dim, num_steps, dim_mults=(1, 2, 4, 8), groups=8,
                 dropout=0.0, classes_guidance=None, padding=False, pad_token=4,
                 heads=4, dim_head=32,K= 2, absolute = False):
        super().__init__()
        self.pad_token = pad_token
        self.padding = padding
        self.dim = dim
        # Embedding (padding-aware if enabled)
        vocab = num_classes + (1 if padding else 0)
        self.K = K
        self.absolute=absolute

        self.embed = nn.Embedding(vocab, dim, padding_idx=(pad_token if padding else None))


        self.abs_pos = AbsPosCache1D(pos_dim=self.dim)
        self.pos_alpha = nn.Parameter(torch.tensor(0.0))

        self.input_pair_router = PairRouter1D(dim=self.dim, dist_feat_dim=2 + K * 2, gate_init=0.0)  # K=2 → 6 ch
        self.final_pair_router = PairRouter1D(dim=self.dim, dist_feat_dim=2 + K * 2, gate_init=0.0)

        # time / label conditioning
        self.time_pos_emb = SinusoidalPosEmb(dim, num_steps=num_steps)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), Mish(), nn.Linear(4 * dim, dim))
        self.label_emb = nn.Embedding(classes_guidance, dim) if classes_guidance is not None else None

        self.dropout = nn.Dropout(p=dropout)

        # channel schedule
        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # Down path

        self.downs = nn.ModuleList([])
        for idx, (dim_in, dim_out) in enumerate(in_out):
            is_last = idx >= (len(in_out) - 1)
            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_out, time_emb_dim=dim, groups=groups),
                #edit
                #FiLM1D(dim_in),
                ResnetBlock(dim_out, dim_out, time_emb_dim=dim, groups=groups),
                Residual(Rezero(LinearAttention1D(dim_out, heads=heads, dim_head=dim_head)) if not is_last else TruePairwiseAttention1D(dim_out, heads=heads, dim_head=dim_head)),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        # Mid (true pairwise)
        mid_dim = dims[-1]
        self.mid_res1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim, groups=groups)
        self.mid_attn = Residual(Rezero(TruePairwiseAttention1D(mid_dim, heads=heads, dim_head=dim_head)))
        self.mid_res2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim, groups=groups)

        # Up path
        self.ups = nn.ModuleList([])
        for idx, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = idx >= (len(in_out) - 1)
            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_in, time_emb_dim=dim, groups=groups),  # concat first
                #edit
                #FiLM1D(dim_in),
                ResnetBlock(dim_in, dim_in, time_emb_dim=dim, groups=groups),
                Residual(Rezero(LinearAttention1D(dim_in, heads=heads, dim_head=dim_head))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))
        #mod_1
        self.final_pair_attn = Residual(
            Rezero(TruePairwiseAttention1D(dims[1], heads=heads, dim_head=dim_head))
        )
        # Head: map back to class logits per position
        self.head = nn.Sequential(
            nn.Conv1d(dims[1], dim, kernel_size=1),
            nn.GroupNorm(1, dim),
            Mish(),
            nn.Conv1d(dim, num_classes, kernel_size=1)
        )

    # --- helpers for masks / guidance pyramids ---

    @staticmethod
    def _mask_guidance(g, m):
        if g is None or m is None:
            return g
        M = (m[:, :, None] & m[:, None, :])  # (B,L,L)
        return g * M.float()

    @staticmethod
    def _pyr1d(mask, n):
        if mask is None:
            return [None] * (n + 1)
        xs = [mask.unsqueeze(1).float()]  # (B,1,L)
        for _ in range(n):
            xs.append(F.max_pool1d(xs[-1], kernel_size=2, stride=2, ceil_mode=True))
        return [x.squeeze(1) > 0.5 for x in xs]  # bool

    @staticmethod
    def _pyr2d(g, n):
        if g is None:
            return [None] * (n + 1)
        xs = [g.unsqueeze(1).float()]  # (B,1,L,L)
        for _ in range(n):
            xs.append(F.max_pool2d(xs[-1], kernel_size=2, stride=2, ceil_mode=True))
        return [x.squeeze(1) for x in xs]  # (B,Ls,Ls)

    # --- forward ---

    def forward(self, time, x, guidance=None, classes=None):
        # x: (B,L) with pad_token used if self.padding
        B, L0 = x.shape
        pad_mask0 = (x != self.pad_token) if self.padding else None

        # clean guidance at full-res (zero pad rows/cols) and build pyramids
        '''
        if guidance is not None and self.padding:
            guidance = self._mask_guidance(guidance, pad_mask0)
        '''
        n_down = len(self.downs)-1
        g_scales = self._pyr2d(guidance, n_down) if guidance is not None else [None] * (n_down + 1)
        m_scales = self._pyr1d(pad_mask0, n_down) if pad_mask0 is not None else [None] * (n_down + 1)

        '''
        #film scale
        deg = g_scales[s].sum(-1)  # (B, L_s)
        deg_norm = deg / (deg.amax(dim=-1, keepdim=True).clamp_min(1.0))
        '''
        # embed tokens
        x = self.embed(x)  # (B,L,D)
        #print(x[0])


        if pad_mask0 is not None:
            x = x * pad_mask0.unsqueeze(-1).float()
        #print(x[0])
        pe = self.abs_pos(L0, x.device).expand(B, -1, -1)
        pe = pe.transpose(1,2)
        pe = pe * pad_mask0.unsqueeze(-1).float()
        x = x + (self.pos_alpha * pe)
        x = x.transpose(1, 2)  # (B,D,L)

        # if you have cm in batch, convert; else build from dot-bracket upstream
        if guidance is not None:
            #since RNAfold version has a potential negative values, this is necessary to make partner work properly
            partner_idx = contactmap_to_partner_idx_1d((guidance > 0).float(), pad_mask0)  # (B,L), −1 on unpaired/pad
        else:
            partner_idx = torch.full((B, L0), -1, dtype=torch.long, device=x_tok.device)

        dist_feats = signed_distance_feats_1d(partner_idx, pad_mask0, K=self.K,absolute=self.absolute)

        x = self.input_pair_router(x, partner_idx, pad_mask0, dist_feats)

        # time / label
        t = self.mlp(self.time_pos_emb(time))
        if classes is not None and self.label_emb is not None:
            t = t + self.label_emb(classes)

        # Down path
        h = []
        #edit
        for i, (res1, res2, attn, down) in enumerate(self.downs):
            x = res1(x, t)
            x = self.dropout(x)
            #edit
            #x = film(x,deg_norm=deg_norm[i])
            x = res2(x, t)
            x = attn(x, guidance=g_scales[i], pad_mask=m_scales[i])  # residualized inside
            h.append(x)  # save BEFORE downsample
            x = down(x)  # last stage is Identity

        # Mid (true pairwise)
        x = self.mid_res1(x, t)
        x = self.mid_attn(x, guidance=g_scales[-1], pad_mask=m_scales[-1])
        x = self.mid_res2(x, t)

        # Up path
        for j, (res1, res2, attn, up) in enumerate(self.ups):
            x = torch.cat([x, h.pop()], dim=1)  # concat at current scale
            x = res1(x, t)
            x = self.dropout(x)
            #level = n_eff - j  # current scale length index
            #x = film(x, deg_norm=g_deg_norm_scales[level])
            x = res2(x, t)
            # choose the matching scale index you used on the way down (mirror):
            level = len(self.downs) - 1 - j
            x = attn(x, guidance=g_scales[level], pad_mask=m_scales[level])
            x = up(x)
        x = self.final_pair_router(x, partner_idx, pad_mask0, dist_feats)
        x = self.final_pair_attn(x, guidance=g_scales[0], pad_mask=m_scales[0])
        logits = self.head(x)  # (B, num_classes, L)
        return logits



class SegmentationUnet1D_inpaint(SegmentationUnet1D):
    """
    1D UNet that conditions on an inpainting mask.

    Convention:
      inpaint_mask: (B, L) float/bool, where 1 = fixed, 0 = free.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # "dedicated channel" adaptation:
        # embed tokens -> (B,L,D), concat mask -> (B,L,D+1), project back -> (B,L,D)
        #self.inpaint_proj = nn.Linear(self.dim + 1, self.dim)

    def forward(self, time, x, guidance=None, classes=None, inpaint_mask=None):
        # x: (B,L) token ids
        B, L0 = x.shape

        pad_mask0 = (x != self.pad_token) if self.padding else None

        # build guidance/pad pyramids
        n_down = len(self.downs) - 1
        g_scales = self._pyr2d(guidance, n_down) if guidance is not None else [None] * (n_down + 1)
        m_scales = self._pyr1d(pad_mask0, n_down) if pad_mask0 is not None else [None] * (n_down + 1)

        # ---- token embedding ----
        x = self.embed(x)  # (B,L,D)

        # ---- inpaint conditioning (dedicated channel -> projection) ----
        if inpaint_mask is not None:
            # expect 0/1 (or bool); make it (B,L,1) float
            m = (inpaint_mask > 0.5).float().unsqueeze(-1)  # (B,L,1)

            # never treat padding as fixed
            if pad_mask0 is not None:
                m = m * pad_mask0.unsqueeze(-1).float()

            #x = torch.cat([x, m], dim=-1)  # (B,L,D+1)
            #x = self.inpaint_proj(x)       # (B,L,D)

        # ---- padding (keep as in base, but make it safe when padding=False) ----
        if pad_mask0 is not None:
            x = x * pad_mask0.unsqueeze(-1).float()

        if pad_mask0 is not None:
            x = x * pad_mask0.unsqueeze(-1).float()
        #print(x[0])
        pe = self.abs_pos(L0, x.device).expand(B, -1, -1)
        pe = pe.transpose(1,2)
        pe = pe * pad_mask0.unsqueeze(-1).float()
        x = x + (self.pos_alpha * pe)
        x = x.transpose(1, 2)  # (B,D,L)

        # ---- partner features / routing ----
        if guidance is not None:
            partner_idx = contactmap_to_partner_idx_1d(guidance, pad_mask0)  # (B,L), -1 on unpaired/pad
        else:
            partner_idx = torch.full((B, L0), -1, dtype=torch.long, device=x.device)

        dist_feats = signed_distance_feats_1d(
            partner_idx,
            pad_mask0,
            K=self.K,
            absolute=self.absolute
        )

        x = self.input_pair_router(x, partner_idx, pad_mask0, dist_feats)

        # ---- time / label conditioning ----
        t = self.mlp(self.time_pos_emb(time))
        if classes is not None and self.label_emb is not None:
            t = t + self.label_emb(classes)

        # ---- Down path ----
        h = []
        for i, (res1, res2, attn, down) in enumerate(self.downs):
            x = res1(x, t)
            x = self.dropout(x)
            x = res2(x, t)
            x = attn(x, guidance=g_scales[i], pad_mask=m_scales[i])
            h.append(x)
            x = down(x)

        # ---- Mid ----
        x = self.mid_res1(x, t)
        x = self.mid_attn(x, guidance=g_scales[-1], pad_mask=m_scales[-1])
        x = self.mid_res2(x, t)

        # ---- Up path ----
        for j, (res1, res2, attn, up) in enumerate(self.ups):
            x = torch.cat([x, h.pop()], dim=1)
            x = res1(x, t)
            x = self.dropout(x)
            x = res2(x, t)

            level = len(self.downs) - 1 - j
            x = attn(x, guidance=g_scales[level], pad_mask=m_scales[level])
            x = up(x)

        # ---- Final routing + head ----
        x = self.final_pair_router(x, partner_idx, pad_mask0, dist_feats)
        x = self.final_pair_attn(x, guidance=g_scales[0], pad_mask=m_scales[0])
        logits = self.head(x)  # (B, num_classes, L)
        return logits