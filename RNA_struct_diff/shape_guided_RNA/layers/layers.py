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


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


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

class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose2d(dim, dim, (1,4), (1,2), 1)

    def forward(self, x):
        return self.conv(x)

class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, (1,3), (1,2), 1)

    def forward(self, x):
        return self.conv(x)

class Rezero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.zeros(1))

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) * self.g

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.GroupNorm(groups, dim_out),
            Mish()
        )
    def forward(self, x):
        return self.block(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            Mish(),
            nn.Linear(time_emb_dim, dim_out)
        )

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        h += self.mlp(time_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)


'''
class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)
        self.contact_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x,contact_map=None):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)

        if contact_map is not None:
            # contact_map: (b, h, w) → flatten to match (h*w)
            contact_bias = contact_map.view(b, 1, 1, h * w)  # (B, 1, 1, N)
            contact_bias = contact_bias.expand(-1, self.heads, -1, -1)  # (B, H, 1, N)
            k = k + self.contact_scale * contact_bias  # broadcasted add

        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)
        self.contact_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x,contact_map=None):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)

        if contact_map is not None:
            # contact_map: (b, h, w) → flatten to match (h*w)
            contact_bias = contact_map.view(b, 1, 1, h * w)  # (B, 1, 1, N)
            contact_bias = contact_bias.expand(-1, self.heads, -1, -1)  # (B, H, 1, N)
            k = k + self.contact_scale * contact_bias  # broadcasted add

        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        hidden_dim = dim_head * heads

        # Projections: input -> Q, K, V
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, kernel_size=1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, kernel_size=1)

        # Learnable scaling for contact map bias
        self.contact_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, contact_map=None):
        b, c, h, w = x.shape
        n = h * w

        # Project and split into Q, K, V
        qkv = self.to_qkv(x)  # (B, 3 * hidden_dim, H, W)
        q, k, v = rearrange(qkv, 'b (qkv heads d) h w -> qkv b heads d (h w)', qkv=3, heads=self.heads)

        # Transpose to (B, H, N, D) for attention computation
        q = q.transpose(-2, -1)  # (B, H, N, D)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        # Scaled dot-product attention logits
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.dim_head ** 0.5)  # (B, H, N, N)

        # Inject pairwise bias from contact map
        if contact_map is not None:
            contact_bias = contact_map.view(b, 1, n, n)  # (B, 1, N, N)
            scores = scores + self.contact_scale * contact_bias  # broadcast over heads

        attn = scores.softmax(dim=-1)  # (B, H, N, N)

        # Weighted sum of values
        out = torch.matmul(attn, v)  # (B, H, N, D)
        out = out.transpose(-2, -1)  # (B, H, D, N)

        # Merge heads and reshape to (B, hidden_dim, H, W)
        out = rearrange(out, 'b heads d (h w) -> b (heads d) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)
'''

from einops import rearrange

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        hidden_dim = heads * dim_head

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, kernel_size=1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, kernel_size=1)

        self.contact_scale = nn.Parameter(torch.tensor(0.1))

    def kernel_fn(self, x):
        return F.elu(x) + 1  # Performer-style positive feature map

    def forward(self, x, contact_map=None):
        b, _, h, w = x.shape
        n = h * w

        # Project to Q, K, V
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv h d) h1 w1 -> qkv b h d (h1 w1)', qkv=3, h=self.heads)

        # Apply kernel
        q = self.kernel_fn(q)
        k = self.kernel_fn(k)

        # Optional: per-position contact guidance (not pairwise!)
        if contact_map is not None:
            if contact_map.dim() == 3:  # (B, H, W)
                contact_map = contact_map.unsqueeze(1)
            assert contact_map.shape == (b, 1, h, w), "Contact map must be (B, 1, H, W)"
            cm = rearrange(contact_map, 'b 1 h w -> b 1 1 (h w)')
            k = k * (1 + self.contact_scale * cm)  # bias key vectors

        # Compute linear attention
        kv = torch.einsum('b h d n, b h e n -> b h d e', k, v)       # (B, H, D, D)
        z = (torch.einsum('b h d n, b h d -> b h n', q, k.sum(dim=-1)) + 1e-6).reciprocal()
        # (B, H, N)
        out = torch.einsum('b h d e, b h n d -> b h n e', kv, q.transpose(-2, -1) ) * z.unsqueeze(-1)  # (B, H, N, D)

        # Reshape and project out
        out = rearrange(out, 'b h (h1 w1) d -> b (h d) h1 w1', h1=h, w1=w)
        return self.to_out(out)

class SegmentationUnet_pairwise(nn.Module):
    def __init__(self, num_classes, dim, num_steps, dim_mults=(1, 2, 4, 8), groups=8,
                 dropout=0., classes_guidance=11, padding=False, pad_token=4,project_contact_map=True):
        super().__init__()

        self.embedding_dim = dim  # original user-specified latent dim (e.g., 64)
        self.embed_dim = dim - 1  # per-token embedding
        self.pad_token = pad_token
        self.padding = padding
        self.project_contact_map = project_contact_map
        pairwise_channels = 2 * self.embed_dim + 1  # [Xi || Xj] + contact map
        if pairwise_channels % groups != 0:
            pairwise_channels += 1  # padding to make GroupNorm happy

        self.dim = pairwise_channels  # CNN input channels (e.g., 128)

        # Embedding layer (padding-aware if needed)
        if padding:
            self.embedding = nn.Embedding(num_classes + 1, self.embed_dim, padding_idx=pad_token)
        else:
            self.embedding = nn.Embedding(num_classes, self.embed_dim)

        self.dropout = nn.Dropout(p=dropout)
        if project_contact_map:
            self.contact_proj = nn.Conv2d(1, 1, kernel_size=1)
        # Time + label conditioning
        self.time_pos_emb = SinusoidalPosEmb(self.embedding_dim, num_steps=num_steps)
        self.mlp = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim * 4),
            Mish(),
            nn.Linear(self.embedding_dim * 4, self.embedding_dim)
        )
        if classes_guidance is not None:
            self.label_emb = nn.Embedding(classes_guidance, self.embedding_dim)

        # CNN base = self.dim, scales with dim_mults
        dims = [self.dim, *map(lambda m: self.dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_out, time_emb_dim=self.embedding_dim),
                ResnetBlock(dim_out, dim_out, time_emb_dim=self.embedding_dim),
                Residual(Rezero(LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=self.embedding_dim)
        self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=self.embedding_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_in, time_emb_dim=self.embedding_dim),
                ResnetBlock(dim_in, dim_in, time_emb_dim=self.embedding_dim),
                Residual(Rezero(LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        # Final projection to go from CNN output channels back to embedding_dim for classification.
        # Yes, we're doing Conv1d after Conv2d because we collapse j-axis — it's stupid, but it works.
        # Could bake this into the last ResNetBlock, but this is easier to reason about.
        self.final_proj = nn.Conv1d(self.dim, self.embedding_dim, kernel_size=1)
        self.final_conv = nn.Sequential(
            nn.Conv1d(self.embedding_dim, self.embedding_dim, kernel_size=1),
            nn.GroupNorm(1, self.embedding_dim),
            Mish(),
            nn.Conv1d(self.embedding_dim, num_classes, kernel_size=1)
        )
    def _inject_contact(self, x, contact_map):
        if contact_map is None:
            return x, None
        B, _, H, W = x.shape
        cm = F.interpolate(contact_map.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False)  # (B, 1, H, W)
        #this uses a CNN to learn features. Could be removed
        if self.project_contact_map:
            cm = self.contact_proj(cm)  # (B, 1, H, W)
        x = x + cm  # adding interpolate
        return x, cm.squeeze(1)  # soft bias for attention: (B, H, W)

    '''
    def __init__(self, num_classes, dim, num_steps, dim_mults=(1, 2, 4, 8), groups = 8, dropout=0., classes_guidance = 11, padding=False):
    
        super().__init__()
        self.padding = padding
        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
    
        #if embedding = True, the embedding class is to be expected to be num_classes
        if padding:
            self.embedding = nn.Embedding(num_classes+1, dim,padding_idx=num_classes)
        else:
            self.embedding = nn.Embedding(num_classes, dim)
        self.dim = dim
        self.num_classes = num_classes
    
        self.dropout = nn.Dropout(p=dropout)
    
        self.time_pos_emb = SinusoidalPosEmb(dim, num_steps=num_steps)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            Mish(),
            nn.Linear(dim * 4, dim)
        )
        if classes_guidance is not None:
            self.label_emb = nn.Embedding(classes_guidance, dim)
    
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
    
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
    
            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_out, time_emb_dim = dim),
                ResnetBlock(dim_out, dim_out, time_emb_dim = dim),
                Residual(Rezero(LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))
    
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = dim)
        self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = dim)
    
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
    
            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_in, time_emb_dim = dim),
                ResnetBlock(dim_in, dim_in, time_emb_dim = dim),
                Residual(Rezero(LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))
        #both for paddign and non padding, since the multinominal class does not "see" the padding class
        out_dim = num_classes
        # out_dim = 1
    
        # Final classification head for per-token discrete diffusion output
        # Input: (B, dim, L), Output: (B, num_classes, L)
        # Note: We use Conv1d here instead of Conv2d because we have collapsed
        # the pairwise matrix along one axis (e.g., mean over j), so we're back
        # to a sequence of L tokens, each needing a class distribution.
        # previously skipped by the unsqueeze/squeeze shenanigan, but this way probably cleaner
        self.final_conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.GroupNorm(1, dim),  # 1 group = LayerNorm-like, safe for Conv1d
            Mish(),
            nn.Conv1d(dim, self.num_classes, kernel_size=1)
        )
    
    
    def forward(self, time, x, classes = None):
        x_shape = x.size()[1:]
    
        ##add modification to handle RNA data by 2D CNN
        if len(x.size()) == 2:
            #first unsqueeze --> channel, second unsqueeze --> simulate 2D image
            x = x.unsqueeze(1).unsqueeze(2)
    
        B, C, H, W = x.size()
        x = self.embedding(x)
    
        assert x.shape == (B, C, H, W, self.dim)
    
        x = x.permute(0, 1, 4, 2, 3)
    
        assert x.shape == (B, C, self.dim, H, W)
    
        x = x.reshape(B, C * self.dim, H, W)
        t = self.time_pos_emb(time)
        t = self.mlp(t)
        if classes is not None:
            t = t + self.label_emb(classes)
    
        h = []
    
        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)
            x = self.dropout(x)
            x = resnet2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)
    
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
    
        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            x = upsample(x)
    
        x= self.final_conv(x)
        x = x.squeeze(2)
        return x
    '''

    def forward(self, time, x, guidance=None, classes=None):
        B, L = x.shape

        x_embed = self.embedding(x)  # (B, L, d-1)

        x_i = x_embed.unsqueeze(2).expand(-1, -1, L, -1)
        x_j = x_embed.unsqueeze(1).expand(-1, L, -1, -1)
        pairwise = torch.cat([x_i, x_j], dim=-1)  # (B, L, L, 2*(d-1))

        # Optional: zero out invalid combinations
        if self.padding:
            token_mask = (x != self.pad_token)  # (B, L)
            pairwise_mask = token_mask.unsqueeze(2) & token_mask.unsqueeze(1)  # (B, L, L)
            pairwise = pairwise * pairwise_mask.unsqueeze(-1).float()

        x = pairwise.permute(0, 3, 1, 2)  # → (B, 2*(d-1), L, L)

        if guidance is not None:
            x = torch.cat([x, guidance.unsqueeze(1)], dim=1)  # +1 channel

        # Pad if needed
        if x.shape[1] != self.dim:
            pad_channels = self.dim - x.shape[1]
            pad = torch.zeros(B, pad_channels, L, L, dtype=x.dtype, device=x.device)
            x = torch.cat([x, pad], dim=1)

        # Time embedding
        t = self.time_pos_emb(time)
        t = self.mlp(t)
        if classes is not None and hasattr(self, 'label_emb'):
            t = t + self.label_emb(classes)

        h = []

        for resnet1, resnet2, attn, downsample in self.downs:
            x, mapped = self._inject_contact(x, guidance)
            x = resnet1(x, t)
            x = self.dropout(x)
            x = resnet2(x, t)
            x = attn(x,mapped)
            h.append(x)
            x = downsample(x)

        x, mapped = self._inject_contact(x, guidance)
        x = self.mid_block1(x, t)
        x = self.mid_attn(x,mapped)
        x = self.mid_block2(x, t)

        for resnet1, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x, mapped = self._inject_contact(x, guidance)
            x = resnet1(x, t)
            x = resnet2(x, t)
            x = attn(x,mapped)
            x = upsample(x)

        x = x.mean(dim=3)  # (B, C, L)
        x = self.final_proj(x)  # (B, embedding_dim, L)
        x = self.final_conv(x)  # (B, num_classes, L)
        return x


class SegmentationUnet_masked(SegmentationUnet_pairwise):
    def __init__(self, num_classes, dim, num_steps, dim_mults=(1, 2, 4, 8), groups=8,
                 dropout=0., classes_guidance=11, padding=False, pad_token=4,project_contact_map=True):
        SegmentationUnet_pairwise.__init__(
            self,
            num_classes=num_classes,
            dim=dim,
            num_steps=num_steps,
            dim_mults=dim_mults,
            groups=groups,
            dropout=dropout,
            classes_guidance=classes_guidance,
            padding=padding,
            pad_token=pad_token,
            project_contact_map=project_contact_map
        )

        self.embedding_dim = dim  # original user-specified latent dim (e.g., 64)
        self.embed_dim = dim - 1  # per-token embedding
        self.pad_token = pad_token
        self.padding = padding
        self.project_contact_map = project_contact_map
        pairwise_channels = 2 * self.embed_dim + 2 # [Xi || Xj] + contact map
        if pairwise_channels % groups != 0:
            pairwise_channels += groups - (pairwise_channels % groups)
        self.dim = pairwise_channels  # CNN input channels (e.g., 128)

        # Embedding layer (padding-aware if needed)
        if padding:
            self.embedding = nn.Embedding(num_classes + 1, self.embed_dim, padding_idx=pad_token)
        else:
            self.embedding = nn.Embedding(num_classes, self.embed_dim)

        self.dropout = nn.Dropout(p=dropout)
        if project_contact_map:
            self.contact_proj = nn.Conv2d(1, 1, kernel_size=1)
        # Time + label conditioning
        self.time_pos_emb = SinusoidalPosEmb(self.embedding_dim, num_steps=num_steps)
        self.mlp = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim * 4),
            Mish(),
            nn.Linear(self.embedding_dim * 4, self.embedding_dim)
        )
        if classes_guidance is not None:
            self.label_emb = nn.Embedding(classes_guidance, self.embedding_dim)

        # CNN base = self.dim, scales with dim_mults
        dims = [self.dim, *map(lambda m: self.dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_out, time_emb_dim=self.embedding_dim),
                ResnetBlock(dim_out, dim_out, time_emb_dim=self.embedding_dim),
                Residual(Rezero(LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=self.embedding_dim)
        self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=self.embedding_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_in, time_emb_dim=self.embedding_dim),
                ResnetBlock(dim_in, dim_in, time_emb_dim=self.embedding_dim),
                Residual(Rezero(LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        # Final projection to go from CNN output channels back to embedding_dim for classification.
        # Yes, we're doing Conv1d after Conv2d because we collapse j-axis — it's stupid, but it works.
        # Could bake this into the last ResNetBlock, but this is easier to reason about.
        self.final_proj = nn.Conv1d(self.dim, self.embedding_dim, kernel_size=1)
        self.final_conv = nn.Sequential(
            nn.Conv1d(self.embedding_dim, self.embedding_dim, kernel_size=1),
            nn.GroupNorm(1, self.embedding_dim),
            Mish(),
            nn.Conv1d(self.embedding_dim, num_classes, kernel_size=1)
        )

    def forward(self, time, x, guidance=None, classes=None, inpaint_mask=None):
        B, L = x.shape

        x_embed = self.embedding(x)  # (B, L, d-1)
        x_i = x_embed.unsqueeze(2).expand(-1, -1, L, -1)
        x_j = x_embed.unsqueeze(1).expand(-1, L, -1, -1)
        pairwise = torch.cat([x_i, x_j], dim=-1)  # (B, L, L, 2*(d-1))
        '''
        # Optional: zero out invalid combinations
        if self.padding:
            token_mask = (x != self.pad_token)  # (B, L)
            pairwise_mask = token_mask.unsqueeze(2) & token_mask.unsqueeze(1)  # (B, L, L)
            pairwise = pairwise * pairwise_mask.unsqueeze(-1).float()
        '''
        if self.padding:
            token_mask = (x != self.pad_token)  # (B,L)
            pairwise_mask = token_mask.unsqueeze(2) & token_mask.unsqueeze(1)  # (B,L,L)
            pairwise = pairwise * pairwise_mask.unsqueeze(-1).float()

        # 3) Permute into CNN format
        x = pairwise.permute(0, 3, 1, 2)  # (B,C,L,L)

        # 4) Add inpainting mask and guidance as extra channels
        if inpaint_mask is not None:
            is_fixed = (inpaint_mask != -1).float()  # (B,L)
            mi = is_fixed.unsqueeze(2).expand(-1, -1, L)
            mj = is_fixed.unsqueeze(1).expand(-1, L, -1)
            mask_pairwise = (mi + mj) / 2.0  # (B,L,L)
            x = torch.cat([x, mask_pairwise.unsqueeze(1)], dim=1)  # add +1 channel

        if guidance is not None:
            x = torch.cat([x, guidance.unsqueeze(1)], dim=1)  # add +1 channel
        # Pad if needed
        if x.shape[1] != self.dim:
            pad_channels = self.dim - x.shape[1]
            pad = torch.zeros(B, pad_channels, L, L, dtype=x.dtype, device=x.device)
            x = torch.cat([x, pad], dim=1)

        # Time embedding
        t = self.time_pos_emb(time)
        t = self.mlp(t)
        if classes is not None and hasattr(self, 'label_emb'):
            t = t + self.label_emb(classes)

        h = []

        for resnet1, resnet2, attn, downsample in self.downs:
            x, mapped = self._inject_contact(x, guidance)
            x = resnet1(x, t)
            x = self.dropout(x)
            x = resnet2(x, t)
            x = attn(x, mapped)
            h.append(x)
            x = downsample(x)

        x, mapped = self._inject_contact(x, guidance)
        x = self.mid_block1(x, t)
        x = self.mid_attn(x, mapped)
        x = self.mid_block2(x, t)

        for resnet1, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x, mapped = self._inject_contact(x, guidance)
            x = resnet1(x, t)
            x = resnet2(x, t)
            x = attn(x, mapped)
            x = upsample(x)

        x = x.mean(dim=3)  # (B, C, L)
        x = self.final_proj(x)  # (B, embedding_dim, L)
        x = self.final_conv(x)  # (B, num_classes, L)
        return x