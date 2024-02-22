import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from model.clip_model.model import load_download_clip, Transformer


class MLPLayer(nn.Module):
    """
    LND - LND or ND - ND
    """

    def __init__(self, dim_list, dropout=0., activation='relu'):
        super().__init__()

        if activation == 'relu':
            self.activation_layer = nn.ReLU()
        elif activation == 'gelu':
            self.activation_layer = nn.GELU()
        else:
            pass

        self.mlp = nn.Sequential()

        for i in range(len(dim_list) - 2):
            _in = dim_list[i]
            _out = dim_list[i + 1]

            self.mlp.add_module(f"linear_{i}", nn.Linear(_in, _out))
            self.mlp.add_module(f"activate_{i}", self.activation_layer)
            self.mlp.add_module(f"dropout_{i}", nn.Dropout(p=dropout))

        self.mlp.add_module(f"linear_final", nn.Linear(dim_list[-2], dim_list[-1]))

    def forward(self, x):
        return self.mlp(x)


class ResidualMLPs(nn.Module):
    """
    Residual MLPs
    ***D - ***D
    """

    def __init__(self, org_dim, hidden_dim, dropout=0., num_layers=2, activation='relu'):
        super().__init__()
        self.num_layers = num_layers

        if activation == 'relu':
            self.activation_layer = nn.ReLU()
        elif activation == 'gelu':
            self.activation_layer = nn.GELU()
        else:
            pass

        self.mlps = nn.ModuleList(nn.Sequential(
            nn.Linear(org_dim, hidden_dim),
            self.activation_layer,
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, org_dim),
        ) for i in range(num_layers))

        self.lns = nn.ModuleList(nn.LayerNorm(org_dim) for i in range(num_layers))

    def forward(self, x):
        for i in range(self.num_layers):
            x = x + self.mlps[i](self.lns[i](x))
        return x


class HashingEncoder(nn.Module):
    """
    hashing encoder, linear projection & tach.
    """

    def __init__(self, org_dim, k_bits, ):
        super().__init__()
        self.fc = nn.Linear(org_dim, k_bits)

    def forward(self, x):
        return torch.tanh(self.fc(x))


class HashingDecoder(nn.Module):
    """
    hashing decoder, MLP & tach.
    """

    def __init__(self, org_bit_dim, recon_bit_dim):
        super().__init__()
        self.mlp = MLPLayer(dim_list=[org_bit_dim, recon_bit_dim, recon_bit_dim])

    def forward(self, x):
        return torch.tanh(self.mlp(x))


class HashingModel(nn.Module):
    """
    Hashing model
    """

    def __init__(self, clip_info=None, args=None):
        super().__init__()

        self.dropout = dropout = args.dropout
        self.activation = activation = args.activation
        self.res_mlp_layers = res_mlp_layers = args.res_mlp_layers
        self.auxiliary_bit_dim = auxiliary_bit_dim = args.auxiliary_bit_dim
        self.transformer_layers = transformer_layers = args.transformer_layers
        self.concept_num = concept_num = args.concept_num
        
        clip_embed_dim = clip_info['embed_dim']

        self.k_bits_list = list(map(int, args.k_bits_list.split(",")))  # str -> list

        self.extend_bits_list = []
        self.extend_bits_list.extend(self.k_bits_list)
        self.extend_bits_list.append(self.auxiliary_bit_dim)

        # share weight.
        self.resmlp_i = self.resmlp_t = ResidualMLPs(org_dim=clip_embed_dim, hidden_dim=4 * clip_embed_dim, dropout=dropout, num_layers=res_mlp_layers, activation=activation)
        
        # share weight.
        self.hash_encoders = nn.ModuleList(
            HashingEncoder(org_dim=clip_embed_dim, k_bits=one)
            for one in self.extend_bits_list
        )
        # share weight.
        self.hash_decoders = nn.ModuleList(
            HashingDecoder(one, auxiliary_bit_dim)
            for one in self.k_bits_list
        )
        # share weight.
        self.concept_embedding = nn.Parameter(torch.randn(concept_num, clip_embed_dim))
        self.ln_q = nn.LayerNorm(clip_embed_dim)

        # patch local cross attention layer
        self.patch_local_attn_layer = nn.MultiheadAttention(embed_dim=clip_embed_dim, num_heads=clip_embed_dim // 64)
        self.ln_kvi = nn.LayerNorm(clip_embed_dim)

        # sentence local cross attention layer
        self.word_local_attn_layer = nn.MultiheadAttention(embed_dim=clip_embed_dim, num_heads=clip_embed_dim // 64)
        self.ln_kvt = nn.LayerNorm(clip_embed_dim)

        self.TEi = Transformer(
            width=clip_embed_dim,
            layers=transformer_layers,
            heads=clip_embed_dim // 64,
        )

        self.TEt = Transformer(
            width=clip_embed_dim,
            layers=transformer_layers,
            heads=clip_embed_dim // 64,
        )

    def forward(self, img_tokens, txt_tokens, img_cls, txt_eos, key_padding_mask):
        output_dict = {}
        bz = img_cls.shape[0]

        # local feature transformation
        qi = qt = (self.concept_embedding).unsqueeze(dim=1).repeat(1, bz, 1)
        kvi = img_tokens
        kvt = txt_tokens
        trans_tokens_i, _ = self.patch_local_attn_layer(self.ln_q(qi), self.ln_kvi(kvi), self.ln_kvi(kvi), need_weights=True)
        trans_tokens_t, _ = self.word_local_attn_layer(self.ln_q(qt), self.ln_kvt(kvt), self.ln_kvt(kvt), need_weights=True,
                                            key_padding_mask=key_padding_mask)

        trans_tokens_i, _ = self.TEi(trans_tokens_i)
        trans_tokens_t, _ = self.TEt(trans_tokens_t)

        output_dict['trans_tokens_i'] = F.normalize(trans_tokens_i, dim=-1)
        output_dict['trans_tokens_t'] = F.normalize(trans_tokens_t, dim=-1)

        # local pooling...
        trans_tokens_i = trans_tokens_i.mean(dim=0)
        trans_tokens_t = trans_tokens_t.mean(dim=0)

        # global feature transformation
        res_img_cls = self.resmlp_i(img_cls)
        res_txt_cls = self.resmlp_t(txt_eos)
        output_dict['res_img_cls'] = F.normalize(res_img_cls, dim=-1)
        output_dict['res_txt_cls'] = F.normalize(res_txt_cls, dim=-1)

        # global-local feature fusion
        img_feature = res_img_cls + trans_tokens_i
        txt_feature = res_txt_cls + trans_tokens_t
        
        output_dict['img_cls_hash'] = {}
        output_dict['txt_cls_hash'] = {}

        output_dict['img_cls_hash_recon'] = {}
        output_dict['txt_cls_hash_recon'] = {}

        for i, one in enumerate(self.extend_bits_list):
            img_cls_hash = self.hash_encoders[i](img_feature)
            txt_cls_hash = self.hash_encoders[i](txt_feature)

            output_dict['img_cls_hash'][one] = img_cls_hash
            output_dict['txt_cls_hash'][one] = txt_cls_hash

            if one != self.auxiliary_bit_dim:
                img_cls_hash_recon = self.hash_decoders[i](img_cls_hash)
                txt_cls_hash_recon = self.hash_decoders[i](txt_cls_hash)
                output_dict['img_cls_hash_recon'][one] = img_cls_hash_recon
                output_dict['txt_cls_hash_recon'][one] = txt_cls_hash_recon

        return output_dict


class CMCL(nn.Module):
    def __init__(self, args=None):
        super(CMCL, self).__init__()
        self.args = args
        self.clip, clip_info = load_download_clip(self.args.clip_path)

        # freeze CLIP
        if self.args.is_freeze_clip:
            for n, p in self.clip.named_parameters():
                p.requires_grad = False

        self.hash = HashingModel(clip_info=clip_info, args=args)

    def forward(self, image, text, key_padding_mask):
        img_tokens, _, img_cls = self.clip.encode_image(image)
        txt_tokens, _, new_key_padding_mask, txt_eos = self.clip.encode_text(text, key_padding_mask)
        output_dict = self.hash(img_tokens, txt_tokens, img_cls, txt_eos, new_key_padding_mask)
        return output_dict
