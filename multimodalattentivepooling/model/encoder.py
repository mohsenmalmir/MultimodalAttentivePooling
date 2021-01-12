from typing import Optional
from torch import Tensor
from torch.nn import Module, AdaptiveMaxPool1d
from torch.nn.modules.transformer import TransformerEncoderLayer, TransformerEncoder, LayerNorm

class WordEncoder(Module):
    """
    This is a wrapper around TransformerEncoder from pytorch.
    """
    d_model: int
    nhead: int
    num_layers: int
    dim_feedforward: int
    ropout: float
    activation: str
    def __init__(self, d_model=128, nhead=6, num_layers=4, dim_feedforward=1024, dropout=0.1, activation="relu", d_out=128):
        super(WordEncoder, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        encoder_norm = LayerNorm(d_model)
        self.encoder = TransformerEncoder(encoder_layer, num_layers, encoder_norm)
        self.pool = AdaptiveMaxPool1d(d_out) # this is for convenience, to make channels equal to visual words

    def forward(self,src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None):
        # src: BTD, transpose it to TBD because...
        src = src.transpose(0, 1) # encoder expects TBD
        enc = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        # transpose back to BTD
        enc = enc.transpose(0, 1) # BTD
        pooled = self.pool(enc)
        return pooled

