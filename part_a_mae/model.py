# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F


# Masked AutoEncoder Class
class MaskedAutoEncoder(nn.Module):
    def __init__(
            self,
            patch_dim = 192,
            num_patches = 64,
            enc_emb_dim = 192,
            encoder_layers = 6,
            encoder_heads = 3,
            enc_d_ff = 768,
            dec_emb_dim = 128,
            decoder_layers = 2,
            decoder_heads = 4,
            mask_ratio = 0.75  # 75% masking
    ):
        super().__init__()
        self.num_patches = num_patches
        self.keep_patches = int(round(num_patches * (1 - mask_ratio)))  # 16

        # Patch + positional embeddings
        self.patch_embed = nn.Linear(patch_dim, enc_emb_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, num_patches, enc_emb_dim))

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=enc_emb_dim,
            nhead=encoder_heads,
            dim_feedforward=enc_d_ff,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)
        self.encoder_norm = nn.LayerNorm(enc_emb_dim)

        # Decoder
        self.decoder_embed = nn.Linear(enc_emb_dim, dec_emb_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_emb_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, dec_emb_dim))

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=dec_emb_dim,
            nhead=decoder_heads,
            dim_feedforward=dec_emb_dim * 4,
            batch_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_layers)
        self.decoder_pred = nn.Linear(dec_emb_dim, patch_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)