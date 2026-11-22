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

    # Major innovation of mae
    # Cut computational cost, create supervised learning task
    def random_masking(self,x):
        num_img, len_seq, dim_patch = x.shape

        # Assign random numbers to each 64 patches
        # Sort the random number smallest to largest
        # [:,:16] :- keep all rows but only 0-15 columns
        keep_idx = torch.rand(num_img, len_seq, device=x.device).argsort(dim=1)[:,:self.keep_patches]
        idx = keep_idx.unsqueeze(-1).expand(-1, -1, dim_patch)       # (num_img, 16, dim_patch)
        x_visible = torch.gather(x, 1, idx)

        mask = torch.ones(num_img, len_seq, device=x.device)
        mask.scatter_(1, keep_idx, 0)                        # set kept positions to 0

        return x_visible, mask, keep_idx

    def forward_encoder(self, patches, apply_mask = True):
        # patches: (B, 64, 192), already patchified
        x = self.patch_embed(patches) + self.pos_emb      # (B, 64, 192), pos embed added BEFORE masking

        if apply_mask:
            x, mask, keep_idx = self.random_masking(x)    # x becomes (B, 16, 192)
        else:
            mask, keep_idx = None, None                   # classifier path: keep all 64

        x = self.encoder(x)                               # 6 transformer layers
        x = self.encoder_norm(x)                          # LayerNorm
        return x, mask, keep_idx

    def forward_decoder(self, x, keep_idx):
         # x: (num_img, 16, 192) encoder output, keep_idx: (B, 16)
        num_img = x.shape[0]

        x = self.decoder_embed(x)                                        # (B, 16, 128)

        # start with 64 mask tokens per image, then drop the visible tokens into their original slots
        full = self.mask_token.expand(num_img, self.num_patches, -1).clone()   # (B, 64, 128)
        for n in range(num_img):
            full[n][keep_idx[n]] = x[n]                                  # visible token goes back to its patch position

        full = full + self.decoder_pos_embed                             # tell every token where it sits
        full = self.decoder(full)                                        # 2 transformer layers
        pred = self.decoder_pred(full)                                   # (B, 64, 192)
        return pred

    def forward(self, patches):
        x, mask, keep_idx = self.forward_encoder(patches)
        pred = self.forward_decoder(x, keep_idx)
        loss = ((pred - patches) ** 2).mean(dim=-1)     # (B, 64) error per patch
        loss = (loss * mask).sum() / mask.sum()          # only masked patches count
        return loss, pred, mask
