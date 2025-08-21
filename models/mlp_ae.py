# models/mlp_ae.py

import torch
import torch.nn as nn
from .modules.encoder import MLPEncoder
from .modules.decoder import MLPDecoder

class MLPAE(nn.Module):
    """
    Combine MLPEncoder and MLPDecoder for source domain inputs.
    Forward:
        x_s: (B, C, T)
    Returns:
        e_s: encoder outputs (latent_dim).
        x_s_recon: decoder outputs (B, C, T)
    """
    def __init__(self,
                 # AE
                 enc_in_dim: int,
                 enc_hidden_dims: list,
                 enc_latent_dim: int,
                 dec_latent_dim: int,
                 dec_hidden_dims: list,
                 dec_out_channels: int,
                 dec_out_seq_len: int,
                 ae_dropout: float = 0.0,
                 ae_use_batchnorm: bool = False
                 ):
        super(MLPAE, self).__init__()

        # MLP Encoder
        self.encoder = MLPEncoder(
            in_dim=enc_in_dim,
            hidden_dims=enc_hidden_dims,
            latent_dim=enc_latent_dim,
            dropout=ae_dropout,
            use_batchnorm=ae_use_batchnorm)
        
        # MLP Decoder
        self.decoder = MLPDecoder(
            latent_dim=dec_latent_dim,
            hidden_dims=dec_hidden_dims,
            out_channels=dec_out_channels,
            out_seq_len=dec_out_seq_len,
            dropout=ae_dropout,
            use_batchnorm=ae_use_batchnorm)
        
    def forward(self, x_s: torch.Tensor):
        e_s = self.encoder(x_s)  # (B, latent_dim)

        x_s_recon = self.decoder(e_s)  # (B, C, T)
        
        return e_s, x_s_recon