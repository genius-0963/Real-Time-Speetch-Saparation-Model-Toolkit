"""
Minimal Standalone SepFormer Speech Separation Model
Extracted from SpeechBrain - only PyTorch dependencies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

EPS = 1e-8


# =====================================================================
# Normalization Layers
# =====================================================================

class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization - normalizes over channel and time dims"""
    def __init__(self, channel_size):
        super().__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, 1, channel_size))
        self.beta = nn.Parameter(torch.Tensor(1, 1, channel_size))
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        # y: [B, C, T] or [B, T, C]
        # Normalize over C and T
        mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        var = (torch.pow(y - mean, 2)).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        return self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta


class ChannelwiseLayerNorm(nn.Module):
    """Channel-wise Layer Normalization - normalizes over channel dimension"""
    def __init__(self, channel_size):
        super().__init__()
        # For input [B, C, T], gamma/beta should be [1, C, 1]
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        # y: [B, C, T] where C is channel dimension
        mean = torch.mean(y, dim=1, keepdim=True)
        var = torch.var(y, dim=1, keepdim=True, unbiased=False)
        return self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta


def select_norm(norm, dim, shape=3, eps=1e-8):
    if norm == "gln":
        return GlobalLayerNorm(dim)
    if norm == "cln":
        return ChannelwiseLayerNorm(dim)
    if norm == "ln":
        return nn.GroupNorm(1, dim, eps=eps)
    return nn.BatchNorm1d(dim)


# =====================================================================
# Transformer Components (Simplified)
# =====================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, input_size, max_len=100000):
        super().__init__()
        self.max_len = max_len
        pe = torch.zeros(max_len, input_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, input_size, 2).float() * (-torch.log(torch.tensor(10000.0)) / input_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, :x.size(1), :]


def get_lookahead_mask(x):
    seq_len = x.size(1)
    mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
    return mask


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ffn, dropout, activation, normalize_before, attention_type="regularMHA", causal=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation() if isinstance(activation, type) else activation
        self.normalize_before = normalize_before
        self.causal = causal
        self.attention_type = attention_type

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        if self.normalize_before:
            src2 = self.norm1(src)
            src2 = self.self_attn(src2, src2, src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
            src = src + self.dropout1(src2)
            src2 = self.norm2(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
            src = src + self.dropout2(src2)
        else:
            src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, nhead, d_ffn, d_model, dropout, activation, normalize_before, causal=False, attention_type="regularMHA"):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, d_ffn, dropout, activation, normalize_before, attention_type, causal)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model) if normalize_before else None

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output, None


# =====================================================================
# SepFormer Blocks
# =====================================================================

class SBTransformerBlock_wnormandskip(nn.Module):
    def __init__(self, num_layers, d_model, nhead, d_ffn=2048, dropout=0.1, activation="relu",
                 use_positional_encoding=False, norm_before=False, causal=False,
                 use_norm=True, use_skip=True, norm_type="gln", attention_type="regularMHA"):
        super().__init__()
        self.use_positional_encoding = use_positional_encoding
        self.causal = causal
        self.use_norm = use_norm
        self.use_skip = use_skip

        if activation == "relu":
            activation = nn.ReLU
        elif activation == "gelu":
            activation = nn.GELU
        else:
            raise ValueError("unknown activation")

        self.mdl = TransformerEncoder(
            num_layers=num_layers,
            nhead=nhead,
            d_ffn=d_ffn,
            d_model=d_model,
            dropout=dropout,
            activation=activation,
            normalize_before=norm_before,
            causal=causal,
            attention_type=attention_type,
        )

        if use_norm:
            self.norm = select_norm(norm_type, d_model, shape=3)

        if use_positional_encoding:
            self.pos_enc = PositionalEncoding(input_size=d_model, max_len=100000)

    def forward(self, x):
        src_mask = get_lookahead_mask(x) if self.causal else None

        if self.use_positional_encoding:
            pos_enc = self.pos_enc(x)
            out = self.mdl(x + pos_enc, src_mask=src_mask)[0]
        else:
            out = self.mdl(x, src_mask=src_mask)[0]

        if self.use_norm:
            out = self.norm(out.permute(0, 2, 1)).permute(0, 2, 1)
        if self.use_skip:
            out = out + x
        return out


# =====================================================================
# Resource Efficient Separation Pipeline (RE-SepFormer)
# =====================================================================

class ResourceEfficientSeparationPipeline(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.0,
                 num_blocks=2, segment_size=20, bidirectional=True,
                 mem_type="av", norm_type="gln", seg_model=None, mem_model=None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.segment_size = segment_size
        self.dropout = dropout
        self.num_blocks = num_blocks
        self.mem_type = mem_type
        self.norm_type = norm_type

        assert mem_type in ["hc", "h", "c", "id", "av", None], f"unsupported mem_type: {mem_type}"

        self.seg_model = nn.ModuleList([seg_model for _ in range(num_blocks)])

        if self.mem_type is not None:
            self.mem_model = nn.ModuleList([mem_model for _ in range(num_blocks - 1)])

        self.output_fc = nn.Sequential(nn.PReLU(), nn.Conv1d(input_size, output_size, 1))

    def forward(self, input):
        B, T, D = input.shape
        input, rest = self._padfeature(input)
        input = input.view(B, -1, self.segment_size, D)  # B, S, K, D
        B, S, K, D = input.shape

        assert K == self.segment_size

        output = input.reshape(B * S, K, D)  # BS, K, D

        if self.mem_type == "av":
            hc = torch.zeros(output.shape[0], 1, output.shape[-1], device=output.device)
        else:
            hc = None

        for i in range(self.num_blocks):
            seg_model_type = type(self.seg_model[0]).__name__
            if seg_model_type == "SBTransformerBlock_wnormandskip":
                output = self.seg_model[i](output + hc)  # BS, K, D
            else:
                raise ValueError("Unsupported segment model class")

            if i < (self.num_blocks - 1):
                if self.mem_type == "av":
                    hc = output.mean(1).unsqueeze(0)
                    hc = self.mem_model[i](hc).permute(1, 0, 2)
                else:
                    hc = self.mem_model[i](hc, S)

        output = output.reshape(B, S * K, D)[:, :T, :]
        output = self.output_fc(output.transpose(1, 2)).transpose(1, 2)
        return output

    def _padfeature(self, input):
        B, T, D = input.shape
        rest = self.segment_size - T % self.segment_size
        if rest > 0:
            input = F.pad(input, (0, 0, 0, rest))
        return input, rest


class ResourceEfficientSeparator(nn.Module):
    """RE-SepFormer Mask Network"""
    def __init__(self, input_dim, causal=True, num_spk=2, nonlinear="relu",
                 layer=3, unit=512, segment_size=20, dropout=0.0,
                 mem_type="hc", seg_model=None, mem_model=None):
        super().__init__()
        self.num_spk = num_spk
        self.segment_size = segment_size

        if mem_type not in ("hc", "h", "c", "id", "av", None):
            raise ValueError(f"Not supporting mem_type={mem_type}")

        self.model = ResourceEfficientSeparationPipeline(
            input_size=input_dim,
            hidden_size=unit,
            output_size=input_dim * num_spk,
            dropout=dropout,
            num_blocks=layer,
            bidirectional=(not causal),
            norm_type="cln" if causal else "gln",
            segment_size=segment_size,
            mem_type=mem_type,
            seg_model=seg_model,
            mem_model=mem_model,
        )

        if nonlinear not in ("sigmoid", "relu", "tanh"):
            raise ValueError(f"Not supporting nonlinear={nonlinear}")
        self.nonlinear = {"sigmoid": nn.Sigmoid(), "relu": nn.ReLU(), "tanh": nn.Tanh()}[nonlinear]

    def forward(self, inpt: torch.Tensor):
        # inpt: [B, N, T] -> [B, T, N]
        inpt = inpt.permute(0, 2, 1)
        B, T, N = inpt.shape
        processed = self.model(inpt)  # B, T, N*num_spk
        processed = processed.reshape(B, T, N, self.num_spk)
        masks = self.nonlinear(processed).unbind(dim=3)
        # [num_spk, B, N, T]
        return torch.stack([m.permute(0, 2, 1) for m in masks])


# =====================================================================
# Encoder / Decoder
# =====================================================================

class ConvTasNetEncoder(nn.Module):
    """ConvTasNet-style encoder with 50% overlap"""
    def __init__(self, kernel_size=16, out_channels=256):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv1d = nn.Conv1d(
            in_channels=1, out_channels=out_channels,
            kernel_size=kernel_size, stride=kernel_size // 2, bias=False
        )

    def forward(self, mixture):
        # mixture: [B, T] -> [B, 1, T]
        mixture = mixture.unsqueeze(1)
        conv_out = self.conv1d(mixture)  # [B, N, K]
        mixture_w = F.relu(conv_out)  # [B, N, K]
        return mixture_w.permute(0, 2, 1)  # [B, K, N]


class ConvTasNetDecoder(nn.Module):
    """ConvTasNet-style decoder with overlap-and-add"""
    def __init__(self, kernel_size=16, in_channels=256):
        super().__init__()
        self.kernel_size = kernel_size
        self.basis_signals = nn.Linear(in_channels, kernel_size, bias=False)

    def forward(self, mixture_w, est_mask):
        # mixture_w: [B, K, N], est_mask: [num_spk, B, N, K]
        source_w = mixture_w.unsqueeze(2).repeat(1, 1, est_mask.size(0), 1) * est_mask.permute(1, 3, 0, 2)  # [B, K, C, N]
        source_w = source_w.permute(0, 2, 1, 3)  # [B, C, K, N]
        est_source = self.basis_signals(source_w)  # [B, C, K, L]
        # overlap and add
        est_source = self._overlap_and_add(est_source, self.kernel_size // 2)  # [B, C, T]
        return est_source.permute(0, 2, 1)  # [B, T, C]

    def _overlap_and_add(self, signal, hop_size):
        """Overlap and add for reconstruction"""
        batch, channels, frames, length = signal.shape
        total_length = (frames - 1) * hop_size + length
        result = signal.new_zeros(batch, channels, total_length)
        for i in range(frames):
            start = i * hop_size
            end = start + length
            result[:, :, start:end] += signal[:, :, i, :]
        return result


# =====================================================================
# Main SepFormer Model
# =====================================================================

class SepFormer(nn.Module):
    """
    Minimal SepFormer Speech Separation Model
    
    Args:
        sample_rate: Audio sample rate (8000 or 16000)
        num_spk: Number of speakers to separate (default: 2)
        kernel_size: Encoder kernel size (default: 16 for 8kHz, 32 for 16kHz)
        enc_dim: Encoder dimension (default: 256)
        hidden_dim: Hidden dimension (default: 512)
        num_blocks: Number of RE-SepFormer blocks (default: 3)
        segment_size: Segment size for chunking (default: 20)
        num_heads: Number of attention heads (default: 8)
        ffn_dim: Feed-forward dimension (default: 2048)
        dropout: Dropout rate (default: 0.1)
        causal: Whether to use causal processing (default: False)
        mem_type: Memory type for RE-SepFormer (default: "av")
    
    Input: [B, T] or [B, 1, T] waveform
    Output: [B, T, num_spk] separated waveforms
    """
    
    def __init__(self, sample_rate=8000, num_spk=2, kernel_size=16, enc_dim=256,
                 hidden_dim=512, num_blocks=3, segment_size=20,
                 num_heads=8, ffn_dim=2048, dropout=0.1, causal=False, mem_type="av"):
        super().__init__()
        self.sample_rate = sample_rate
        self.num_spk = num_spk
        self.kernel_size = kernel_size
        self.enc_dim = enc_dim

        # Encoder
        self.encoder = ConvTasNetEncoder(kernel_size=kernel_size, out_channels=enc_dim)

        # Mask Network (RE-SepFormer) - operates directly on enc_dim
        seg_model = SBTransformerBlock_wnormandskip(
            num_layers=1, d_model=enc_dim, nhead=num_heads,
            d_ffn=ffn_dim, dropout=dropout, activation="relu",
            use_positional_encoding=True, norm_before=True,
            causal=causal, use_norm=True, use_skip=True,
            norm_type="cln"  # Use channel-wise layer norm for transformer blocks
        )
        mem_model = SBTransformerBlock_wnormandskip(
            num_layers=1, d_model=enc_dim, nhead=num_heads,
            d_ffn=ffn_dim, dropout=dropout, activation="relu",
            use_positional_encoding=False, norm_before=True,
            causal=causal, use_norm=True, use_skip=True,
            norm_type="cln"  # Use channel-wise layer norm for transformer blocks
        )

        self.masknet = ResourceEfficientSeparator(
            input_dim=enc_dim, causal=causal, num_spk=num_spk,
            nonlinear="relu", layer=num_blocks, unit=hidden_dim,
            segment_size=segment_size, dropout=dropout,
            mem_type=mem_type, seg_model=seg_model, mem_model=mem_model
        )

        # Decoder
        self.decoder = ConvTasNetDecoder(kernel_size=kernel_size, in_channels=enc_dim)

    def forward(self, mix):
        """
        Args:
            mix: [B, T] or [B, 1, T] mixture waveform
        Returns:
            [B, T, num_spk] separated waveforms
        """
        if mix.dim() == 3 and mix.size(1) == 1:
            mix = mix.squeeze(1)  # [B, T]
        
        # Encode
        mix_w = self.encoder(mix)  # [B, K, N]

        # Separate
        est_mask = self.masknet(mix_w.permute(0, 2, 1))  # [num_spk, B, N, K]

        # Decode
        est_source = self.decoder(mix_w, est_mask)  # [B, T, num_spk]

        # Trim/pad to match input length
        T_origin = mix.size(1)
        T_est = est_source.size(1)
        if T_origin > T_est:
            est_source = F.pad(est_source, (0, 0, 0, T_origin - T_est))
        else:
            est_source = est_source[:, :T_origin, :]

        return est_source

    def separate_file(self, audio_path, device="cpu"):
        """Convenience method to separate a WAV file"""
        import torchaudio
        waveform, sr = torchaudio.load(audio_path)
        waveform = waveform.to(device)
        
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate).to(device)
            waveform = resampler(waveform)
        
        self.eval()
        with torch.no_grad():
            waveform = waveform.to(device)
            est_sources = self.forward(waveform)
            # Normalize
            est_sources = est_sources / est_sources.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-8)
        
        return est_sources.cpu()


# =====================================================================
# Factory Functions
# =====================================================================

def sepformer_wham(sample_rate=8000, num_spk=2, **kwargs):
    """SepFormer for WHAM! dataset (8kHz)"""
    return SepFormer(
        sample_rate=sample_rate, num_spk=num_spk,
        kernel_size=16, enc_dim=256, hidden_dim=512,
        num_blocks=3, segment_size=20, num_heads=8,
        ffn_dim=2048, dropout=0.1, causal=False, mem_type="av",
        **kwargs
    )

def sepformer_whamr(sample_rate=8000, num_spk=2, **kwargs):
    """SepFormer for WHAMR! dataset (8kHz)"""
    return SepFormer(
        sample_rate=sample_rate, num_spk=num_spk,
        kernel_size=16, enc_dim=256, hidden_dim=512,
        num_blocks=4, segment_size=20, num_heads=8,
        ffn_dim=2048, dropout=0.1, causal=False, mem_type="av",
        **kwargs
    )

def sepformer_wsj02mix(sample_rate=8000, num_spk=2, **kwargs):
    """SepFormer for WSJ0-2mix dataset (8kHz)"""
    return SepFormer(
        sample_rate=sample_rate, num_spk=num_spk,
        kernel_size=16, enc_dim=256, hidden_dim=512,
        num_blocks=3, segment_size=20, num_heads=8,
        ffn_dim=2048, dropout=0.1, causal=False, mem_type="av",
        **kwargs
    )

def sepformer_librimix(sample_rate=16000, num_spk=2, **kwargs):
    """SepFormer for LibriMix dataset (16kHz)"""
    return SepFormer(
        sample_rate=sample_rate, num_spk=num_spk,
        kernel_size=32, enc_dim=256, hidden_dim=512,
        num_blocks=3, segment_size=20, num_heads=8,
        ffn_dim=2048, dropout=0.1, causal=False, mem_type="av",
        **kwargs
    )


# =====================================================================
# Demo / Test
# =====================================================================

if __name__ == "__main__":
    import time
    
    # Create model
    model = sepformer_wham()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    B, T = 2, 16000  # 2 seconds at 8kHz
    x = torch.randn(B, T)
    
    model.eval()
    with torch.no_grad():
        start = time.time()
        out = model(x)
        elapsed = time.time() - start
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")  # [B, T, num_spk]
    print(f"Inference time: {elapsed*1000:.1f}ms")
    print(f"RTF: {elapsed / (T / 8000):.3f}")  # Real-time factor