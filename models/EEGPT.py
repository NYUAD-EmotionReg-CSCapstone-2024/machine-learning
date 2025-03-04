import math
import torch
import torch.nn as nn

CHANNEL_DICT = {
	k.upper(): v
	for v, k in enumerate([
		'FP1', 'FPZ', 'FP2',
		'AF7', 'AF3', 'AF4', 'AF8',
		'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
		'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
		'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
		'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
		'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
		'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8',
		'O1', 'OZ', 'O2',
	])
}

SEEDV_CHANNELS = [
    'FP1', 'FPZ', 'FP2', 
    'AF3', 'AF4', 
    'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 
    'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 
    'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 
    'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 
    'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 
    'CB1', 'O1', 'OZ', 'O2', 'CB2'
]


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
	# Cut & paste from PyTorch official master until it's in a few official releases - RW
	# Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
	def norm_cdf(x):
		# Computes standard normal cumulative distribution function
		return (1. + math.erf(x / math.sqrt(2.))) / 2.

	with torch.no_grad():
		# Values are generated by using a truncated uniform distribution and
		# then using the inverse CDF for the normal distribution.
		# Get upper and lower cdf values
		l = norm_cdf((a - mean) / std)
		u = norm_cdf((b - mean) / std)
		# Uniformly fill tensor with values from [l, u], then translate to
		# [2l-1, 2u-1].
		tensor.uniform_(2 * l - 1, 2 * u - 1)
		# Use inverse cdf transform for normal distribution to get truncated
		# standard normal
		tensor.erfinv_()
		# Transform to proper mean, std
		tensor.mul_(std * math.sqrt(2.))
		tensor.add_(mean)
		# Clamp to ensure it's in the proper range
		tensor.clamp_(min=a, max=b)
		return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
	# type: (Tensor, float, float, float, float) -> Tensor
	return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def apply_mask(mask, x):
	"""
	:param x: tensor of shape [B (batch-size), N (num-patches), C, D (feature-dim)]
	:param mask: tensor [mN, mC] containing indices of patches in [N, C] to keep
	"""
	B, N, C, D = x.shape
	if len(mask.shape) == 2:
		mN, mC = mask.shape
		mask_keep = mask.reshape((1, mN * mC, 1)).repeat((B, 1, D))
		masked_x = torch.gather(x.reshape((B, N * C, D)), dim=-2, index=mask_keep)
		masked_x = masked_x.contiguous().view((B, mN, mC, D))
	else:
		mN = mask.shape[0]
		mask_keep = mask.reshape((1, mN, 1)).repeat((B, 1, D))
		masked_x = torch.gather(x.reshape((B, N * C, D)), dim=-2, index=mask_keep)
	return masked_x


def apply_mask_t(mask_t, x):
	"""
	:param x: tensor of shape [B (batch-size), N (num-patches), C, D (feature-dim)]
	:param mask: tensor [mN, mC] containing indices of patches in [N, C] to keep
	"""
	B, N, D = x.shape
	mN = mask_t.shape[0]
	mask_keep = mask_t.reshape((1, mN, 1)).repeat((B, 1, D))
	masked_x = torch.gather(x, dim=1, index=mask_keep)
	return masked_x


class RotaryEmbedding(nn.Module):

	def __init__(self, dim, theta=10000, learned_freq=False, interpolate_factor=1):
		super().__init__()
		self.cache = dict()
		self.freqs = nn.Parameter(
			1. / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)),
			requires_grad=learned_freq
		)
		assert interpolate_factor >= 1
		self.interpolate_factor = interpolate_factor

	def prepare_freqs(self, num_patches=(1, 8), device='cuda', dtype=torch.float, offset=0):
		C, N = num_patches
		cache_key = f'freqs:{num_patches}:{device}:{dtype}:{offset}'
		if cache_key in self.cache:
			return self.cache[cache_key]
		seq_pos = torch.arange(N, device=device, dtype=dtype)
		seq_pos = seq_pos.repeat_interleave(repeats=C, dim=0)
		seq_pos = (seq_pos + offset) / self.interpolate_factor
		freqs = self.freqs
		freqs = torch.outer(seq_pos.type(freqs.dtype), freqs)
		freqs = freqs.repeat_interleave(repeats=2, dim=-1)
		self.cache[cache_key] = freqs
		return freqs


class DropPath(nn.Module):
	"""Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

	def __init__(self, drop_prob=None):
		super(DropPath, self).__init__()
		self.drop_prob = drop_prob

	def drop_path(self, x, drop_prob: float = 0., training: bool = False):
		if drop_prob == 0. or not training:
			return x
		keep_prob = 1 - drop_prob
		shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
		random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
		random_tensor.floor_()  # binarize
		output = x.div(keep_prob) * random_tensor
		return output

	def forward(self, x):
		return self.drop_path(x, self.drop_prob, self.training)


def rotate_half(x):
	# x = rearrange(x, '... (d r) -> ... d r', r = 2)
	x = x.reshape((*x.shape[:-1], x.shape[-1] // 2, 2))
	x1, x2 = x.unbind(dim=-1)
	x = torch.stack((-x2, x1), dim=-1)
	# return rearrange(x, '... d r -> ... (d r)')
	return x.flatten(-2)


def apply_rotary_emb(freqs, t, start_index=0, scale=1.):
	freqs = freqs.to(t)
	rot_dim = freqs.shape[-1]
	end_index = start_index + rot_dim
	assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'
	t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
	t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
	return torch.cat((t_left, t, t_right), dim=-1)


class PatchEmbed(nn.Module):
	def __init__(self, img_size, patch_size, patch_stride=None, embed_dim=512):
		super().__init__()
		self.img_size = img_size
		self.patch_size = patch_size
		self.patch_stride = patch_stride
		if patch_stride is None:
			self.num_patches = (img_size[0], img_size[1] // patch_size)
		else:
			# FIX: use patch_size rather than patch_stride to compute number of patches along time dimension
			self.num_patches = (img_size[0], (img_size[1] - patch_size) // patch_stride + 1)
		self.proj = nn.Conv2d(
			1,
			embed_dim,
			kernel_size=(1, patch_size),
			stride=(1, patch_size if patch_stride is None else patch_stride),
		)

	def forward(self, x):
		# x: B, C, T
		x = x.unsqueeze(1)  # B, 1, C, T
		x = self.proj(x).transpose(1, 3)  # B, T, C, D
		return x


class Attention(nn.Module):
	def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., is_causal=False, use_rope=False, return_attention=False):
		super().__init__()
		self.num_heads = num_heads
		self.head_dim = dim // num_heads
		self.use_rope = use_rope
		self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
		self.attn_drop = attn_drop
		self.proj = nn.Linear(dim, dim)
		self.proj_drop = nn.Dropout(proj_drop)
		self.is_causal = is_causal
		self.return_attention = return_attention

	def forward(self, x, freqs=None):
		B, T, C = x.shape
		qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # 3, B, nh, t, d
		q, k, v = qkv[0], qkv[1], qkv[2]  # B, nh, t, d
		if self.use_rope:  # RoPE
			q = apply_rotary_emb(freqs, q)
			k = apply_rotary_emb(freqs, k)
		if self.return_attention:
			if self.is_causal:
				attn_mask = torch.ones(q.size(-2), q.size(-2), dtype=torch.bool).tril(diagonal=0)
				attn_maak = torch.zeros(q.size(-2), q.size(-2))
				attn_mask = attn_maak.masked_fill(torch.logical_not(attn_mask), -float('inf'))
				attn_weight = torch.softmax((q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))) + attn_mask, dim=-1)
			else:
				attn_weight = torch.softmax((q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))), dim=-1)
			return attn_weight
		y = torch.nn.functional.scaled_dot_product_attention(
			q, k, v, attn_mask=None, dropout_p=self.attn_drop if self.training else 0, is_causal=self.is_causal)
		x = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, nh, T, hs) -> (B, T, hs*nh)
		x = self.proj(x)
		x = self.proj_drop(x)
		return x


class MLP(nn.Module):
	def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
		super().__init__()
		out_features = out_features or in_features
		hidden_features = hidden_features or in_features
		self.fc1 = nn.Linear(in_features, hidden_features)
		self.act = act_layer()
		self.fc2 = nn.Linear(hidden_features, out_features)
		self.drop = nn.Dropout(drop)

	def forward(self, x):
		x = self.fc1(x)
		x = self.act(x)
		x = self.drop(x)
		x = self.fc2(x)
		x = self.drop(x)
		return x


class Block(nn.Module):

	def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, is_causal=False, use_rope=False, return_attention=False):
		super().__init__()
		self.return_attention = return_attention
		self.norm1 = norm_layer(dim)
		self.attn = Attention(
			dim,
			num_heads=num_heads,
			qkv_bias=qkv_bias,
			attn_drop=attn_drop,
			proj_drop=drop,
			is_causal=is_causal,
			use_rope=use_rope,
			return_attention=return_attention
		)
		self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
		self.norm2 = norm_layer(dim)
		mlp_hidden_dim = int(dim * mlp_ratio)
		self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

	def forward(self, x, freqs=None):
		y = self.attn(self.norm1(x), freqs)
		if self.return_attention:
			return y
		x = x + self.drop_path(y)
		x = x + self.drop_path(self.mlp(self.norm2(x)))
		return x


class EEGTransformer(nn.Module):
	""" EEG Transformer """
	def __init__(
		self,
		img_size=(62, 1000),
		patch_size=32*2,
		patch_stride=32,
		embed_num=4,
		embed_dim=64,
		depth=8,
		num_heads=8,
		mlp_ratio=4.0,
		qkv_bias=True,
		drop_rate=0.0,
		attn_drop_rate=0.0,
		drop_path_rate=0.0,
		norm_layer=nn.LayerNorm,
		patch_module=PatchEmbed,
		init_std=0.02,
		return_attention_layer=-1,
	):
		super().__init__()
		self.num_features = self.embed_dim = embed_dim
		self.embed_num = embed_num
		self.num_heads = num_heads
		# --
		self.patch_embed = patch_module(
			img_size=img_size,
			patch_size=patch_size,
			patch_stride=patch_stride,
			embed_dim=embed_dim
		)
		self.num_patches = self.patch_embed.num_patches
		# --
		self.chan_embed = nn.Embedding(len(CHANNEL_DICT), embed_dim)
		# --
		dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
		self.blocks = nn.ModuleList([
			Block(
				dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
				drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
				is_causal=False, use_rope=False, return_attention=(i + 1) == return_attention_layer
			)
			for i in range(depth)
		])
		self.norm = norm_layer(embed_dim)
		# ------
		self.init_std = init_std
		self.summary_token = nn.Parameter(torch.zeros(1, embed_num, embed_dim))
		trunc_normal_(self.summary_token, std=self.init_std)
		self.apply(self._init_weights)
		self.fix_init_weight()

	def prepare_chan_ids(self, channels):
		chan_ids = []
		for ch in channels:
			ch = ch.upper().strip('.')
			assert ch in CHANNEL_DICT
			chan_ids.append(CHANNEL_DICT[ch])
		return torch.tensor(chan_ids).unsqueeze_(0).long()

	def fix_init_weight(self):
		def rescale(param, layer_id):
			param.div_(math.sqrt(2.0 * layer_id))
		for layer_id, layer in enumerate(self.blocks):
			rescale(layer.attn.proj.weight.data, layer_id + 1)
			rescale(layer.mlp.fc2.weight.data, layer_id + 1)

	def _init_weights(self, m):
		if isinstance(m, nn.Linear):
			trunc_normal_(m.weight, std=self.init_std)
			if isinstance(m, nn.Linear) and m.bias is not None:
				nn.init.constant_(m.bias, 0)
		elif isinstance(m, nn.LayerNorm):
			nn.init.constant_(m.bias, 0)
			nn.init.constant_(m.weight, 1.0)
		elif isinstance(m, nn.Conv2d):
			trunc_normal_(m.weight, std=self.init_std)
			if m.bias is not None:
				nn.init.constant_(m.bias, 0)
		elif isinstance(m, nn.Embedding):
			torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

	def forward(self, x, chan_ids=None, mask_x=None, mask_t=None):
		# x.shape B, C, T
		# mask_x.shape mN, mC
		# mask_t.shape mN
		# -- patchify x
		x = self.patch_embed(x)  # 
		B, N, C, D = x.shape
		assert N == self.num_patches[1] and C == self.num_patches[0], f"{N}=={self.num_patches[1]} and {C}=={self.num_patches[0]}"
		if chan_ids is None:
			chan_ids = torch.arange(0, C)
		chan_ids = chan_ids.to(x)
		# -- add channels positional embedding to x
		x = x + self.chan_embed(chan_ids.long()).unsqueeze(0)  # (1,C) -> (1,1,C,D)
		if mask_x is not None:
			mask_x = mask_x.to(x.device)
			x = apply_mask(mask_x, x)  # B, mN, mC, D
			B, N, C, D = x.shape
		x = x.flatten(0, 1)  # BmN, mC, D
		# -- concat summary token
		summary_token = self.summary_token.repeat((x.shape[0], 1, 1))
		x = torch.cat([x, summary_token], dim=1)  # BmN, mC+embed_num, D
		# -- fwd prop
		for i, blk in enumerate(self.blocks):
			x = blk(x)  # B*N, mC+1, D
			if blk.return_attention == True:
				return x
		x = x[:, -summary_token.shape[1]:, :]
		if self.norm is not None:
			x = self.norm(x)
		x = x.flatten(-2)
		x = x.reshape((B, N, -1))
		# -- reshape back
		if mask_t is not None:
			mask_t = mask_t.to(x.device)
			x = apply_mask_t(mask_t, x)  # B, mN, D
		x = x.reshape((B, N, self.embed_num, -1))
		return x

class EEGPTEncoder(nn.Module):
	def __init__(
		self, 
		ckpt_path, 
		window=4, 
		freq=256, 
		patch_size=64, 
		patch_stride=32, 
		embed_num=4, 
		embed_dim=512
	):
		super().__init__()

		self.channels_to_keep = [(i, ch) for i, ch in enumerate(SEEDV_CHANNELS) if ch in CHANNEL_DICT]
		self.chan_ids = torch.tensor([CHANNEL_DICT[ch] for _, ch in self.channels_to_keep])
		self.keep_indices = torch.tensor([i for i, _ in self.channels_to_keep])

		self.eegpt = EEGTransformer(
			img_size=(len(self.channels_to_keep), window * freq),
			patch_size=patch_size,
			patch_stride=patch_stride,
			embed_num=embed_num,
			embed_dim=embed_dim,
		)
		self._load_weights(ckpt_path)

		self.out_shape = (self.eegpt.num_patches[1] * embed_num, embed_dim)

	def _load_weights(self, ckpt_path):
		ckpt = torch.load(ckpt_path)
		encoder_stat = {}
		for k, v in ckpt['state_dict'].items():
			if k.startswith("target_encoder."):
				encoder_stat[k[15:]] = v
		self.eegpt.load_state_dict(encoder_stat)
	
	def forward(self, x):
		x = x[:, self.keep_indices, :]
		x = self.eegpt(x, self.chan_ids)
		return x