import os.path as osp
from einops import repeat
import torch
import torch.nn as nn

from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from Dassl.dassl.engine.trainer import TrainerX
from Dassl.dassl.metrics import compute_accuracy
from Dassl.dassl.utils import load_pretrained_weights, load_checkpoint
from Dassl.dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()




def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = {
        "trainer": 'FedTPG',
        "vision_depth": 0,
        "language_depth": 0,
        "vision_ctx": 0,
        "language_ctx": 0
    }

    model = clip.build_model(state_dict or model.state_dict(), design_details)
    return model


def exists(val):
    return val is not None


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x_q, x_kv=None, **kwargs):
        x_q = self.norm(x_q)

        if exists(x_kv):
            x_kv = self.norm_context(x_kv)
        else:
            x_kv = x_q

        return self.fn(x_q, x_kv, x_kv, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)


class CrossAttention(nn.Module):
    def __init__(
            self,
            latent_dim,
            kv_dim,
            cross_heads=4,
            seq_dropout_prob=0.
    ):
        super().__init__()
        self.seq_dropout_prob = seq_dropout_prob

        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(latent_dim,
                    nn.MultiheadAttention(latent_dim, num_heads=cross_heads, kdim=kv_dim, vdim=kv_dim,
                                          dropout=seq_dropout_prob, batch_first=True),
                    context_dim=kv_dim),
            FeedForward(latent_dim)])

    def forward(
            self,
            data,
            soft_prompt,
            mask=None,
    ):
        b, *_, device = *data.shape, data.device
        x = repeat(soft_prompt, 'n d -> b n d', b=b)
        cross_attn, cross_ff = self.cross_attend_blocks
        x, _ = cross_attn(x, data, key_padding_mask=mask)
        x = cross_ff(x)+x

        return x


class SelfAttention(nn.Module):
    def __init__(
            self,
            depth,
            latent_dim,
            latent_heads=4,
    ):
        super().__init__()

        self.layers = nn.ModuleList([])

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(latent_dim, nn.MultiheadAttention(latent_dim, num_heads=latent_heads, batch_first=True)),
                FeedForward(latent_dim)
            ]))

    def forward(
            self,
            x,
            mask=None
    ):
        # layers

        for self_attn, self_ff in self.layers:
            x = self_attn(x, key_padding_mask=mask)[0] + x
            x = self_ff(x) + x
        return x


class PromptTranslator(nn.Module):
    def __init__(
            self,
            prompt_len,
            prompt_depth,
            prompt_dim = 512,
            depth=4,
            self_heads = 4,
            cross_heads = 4,
            textemb_dim=512,
            device='cuda'
    ):
        super().__init__()
        self.device = device
        self.prompt_len = prompt_len
        self.prompt_depth = prompt_depth
        prompt_dim = prompt_dim
        soft_prompt = torch.empty(prompt_len*prompt_depth, prompt_dim)
        nn.init.normal_(soft_prompt, std=0.02)
        self.soft_prompt = nn.Parameter(soft_prompt)

        self.encoder = CrossAttention(
            latent_dim=prompt_dim,
            kv_dim=textemb_dim,
            cross_heads= cross_heads,
        )
        if depth>0:
            self.transformer = SelfAttention(depth=depth, latent_dim=prompt_dim,latent_heads = self_heads)

        # self.vis_linear = nn.Linear(512,768)
        self.depth = depth
    def forward(
            self,
            text_emb,
    ):
        # first output is text prompt, second output is image prompt
        prompt = self.encoder(text_emb, self.soft_prompt)
        if self.depth>0:
            prompt = self.transformer(prompt)
        prompt = prompt.reshape(self.prompt_depth,self.prompt_len,-1)
        # vis_prompt = self.vis_linear(prompt)

        return prompt,prompt


class ImageEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()

        self.conv1 = clip_model.conv1
        self.class_embedding = clip_model.class_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.ln_pre = clip_model.ln_pre
        self.transformer = clip_model.transformer
        self.ln_post = clip_model.ln_post
        self.proj = clip_model.proj

    def forward(self, x, vis_ctx=[]):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, vis_ctx, False)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, text_ctx):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, text_ctx, True)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


# class PromptLearner(nn.Module):
#     def __init__(self, cfg, classnames, clip_model):
#         super().__init__()
#         n_cls = len(classnames)
#         n_ctx = cfg.TRAINER.FedTPG.N_CTX
#         ctx_depth = cfg.TRAINER.FedTPG.D_CTX
#         self.meta_net = PromptTranslator(n_ctx, ctx_depth, depth=cfg.TRAINER.FedTPG.DEPTH)
#         self.meta_net.half()
#
#         self.ctx_depth = ctx_depth
#         self.n_ctx = n_ctx
#         dtype = clip_model.dtype
#         ctx_dim = clip_model.ln_final.weight.shape[0]
#         self.N = cfg.TRAINER.PLOT.N
#
#         ctx_vectors = torch.empty(self.N, n_ctx, ctx_dim, dtype=dtype)
#         nn.init.normal_(ctx_vectors, std=0.02)
#         prompt_prefix = " ".join(["X"] * n_ctx)
#
#         classnames = [name.replace("_", " ") for name in classnames]
#         name_lens = [len(_tokenizer.encode(name)) for name in classnames]
#         prompts = [prompt_prefix + " " + name + "." for name in classnames]
#         tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
#         tokenized_prompts = tokenized_prompts.repeat(self.N, 1)
#
#         with torch.no_grad():
#             embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
#
#         self.register_buffer("token_prefix", embedding[:, :1, :])
#         self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])
#
#         self.ctx = nn.Parameter(ctx_vectors)
#         self.n_cls = n_cls
#         self.n_ctx = n_ctx
#         self.tokenized_prompts = tokenized_prompts
#         self.name_lens = name_lens
#         self.class_token_position = cfg.TRAINER.PLOT.CLASS_TOKEN_POSITION
#
#     def forward(self):
#         ctx = self.ctx
#         print(ctx.shape)
#         ctx = ctx.permute(1, 0, 2, 3).contiguous().view(self.N * self.n_cls, self.n_ctx, ctx.shape[3])
#
#         prefix = self.token_prefix
#         suffix = self.token_suffix
#
#         if self.class_token_position == "end":
#             prompts = torch.cat([prefix, ctx, suffix], dim=1)
#         elif self.class_token_position == "middle":
#             half_n_ctx = self.n_ctx // 2
#             prompts = []
#             for i in range(self.n_cls):
#                 name_len = self.name_lens[i]
#                 prefix_i = prefix[i: i + 1, :, :]
#                 class_i = suffix[i: i + 1, :name_len, :]
#                 suffix_i = suffix[i: i + 1, name_len:, :]
#                 ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
#                 ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
#                 prompt = torch.cat([prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i], dim=1)
#                 prompts.append(prompt)
#             prompts = torch.cat(prompts, dim=0)
#         elif self.class_token_position == "front":
#             prompts = []
#             for i in range(self.n_cls):
#                 name_len = self.name_lens[i]
#                 prefix_i = prefix[i: i + 1, :, :]
#                 class_i = suffix[i: i + 1, :name_len, :]
#                 suffix_i = suffix[i: i + 1, name_len:, :]
#                 ctx_i = ctx[i: i + 1, :, :]
#                 prompt = torch.cat([prefix_i, class_i, ctx_i, suffix_i], dim=1)
#                 prompts.append(prompt)
#             prompts = torch.cat(prompts, dim=0)
#         else:
#             raise ValueError
#
#         return prompts

class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        self.n_cls = n_cls

        n_ctx = cfg.TRAINER.FedTPG.N_CTX
        self.n_ctx = n_ctx

        ctx_depth = cfg.TRAINER.FedTPG.D_CTX
        self.ctx_depth = ctx_depth
        self.N = cfg.TRAINER.PLOT.N

        self.meta_net = PromptTranslator(n_ctx, ctx_depth, depth=cfg.TRAINER.FedTPG.DEPTH)
        self.meta_net.half()


        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        ctx_vectors = torch.empty(self.N, n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        tokenized_prompts = tokenized_prompts.repeat(self.N, 1)

        with torch.no_grad():

            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])

        self.ctx = nn.Parameter(ctx_vectors)

        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.PLOT.CLASS_TOKEN_POSITION

    def forward(self, context_emb):
        text_ctx, vis_ctx = self.meta_net(context_emb.unsqueeze(0))  # (n_ctx, ctx_dim) # self.ctx

        return text_ctx, vis_ctx


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.n_cls = len(classnames)
        self.set_prompt_prefix()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = ImageEncoder(clip_model.visual)
        self.text_encoder = TextEncoder(clip_model)
        self.token_embedding = clip_model.token_embedding
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.device = torch.device("cuda:0")
        self.device1 = torch.device("cuda")
        self.N = cfg.TRAINER.PLOT.N
        self.dataset = cfg.DATASET.NAME
        self.frac = cfg.TRAINER.PROMPTFL.FRAC
        self.clip_model_ = clip_model
        self.classnames = classnames


    def set_prompt_prefix(self):

        n_ctx = self.cfg.TRAINER.FedTPG.N_CTX
        # random initialization
        self.n_ctx = n_ctx
        self.prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{self.prompt_prefix}"')
        print(f"Number of context words (tokens): {self.n_ctx}")

    def get_tokenized_classnames(self, classnames):

        prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = self.token_embedding(tokenized_prompts.to(self.device)).type(self.dtype)
        # token_prefix = embedding[:, :1, :]  # SOS
        # token_suffix = embedding[:, 1 + self.n_ctx:, :]  # CLS, EOS
        return embedding, tokenized_prompts

    def encode_image(self, image, vis_ctx):
        return self.image_encoder(image.type(self.dtype))

    def encode_text(self, classnames, text_features_):

        context_emb = text_features_
        prompt_vectors, tokenized_prompts = self.get_tokenized_classnames(classnames)

        text_ctx, vis_ctx = self.prompt_learner(context_emb)

        prompt_vectors = torch.cat(
            [
                prompt_vectors[:, :1],  # (dim0, 1, dim)
                text_ctx[0].unsqueeze(0).expand(prompt_vectors.shape[0], -1, -1),  # (dim0, n_ctx, dim)
                prompt_vectors[:, 1 + text_ctx.shape[1]:],  # (dim0, *, dim)
            ],
            dim=1,
        )
        if len(text_ctx) > 1:
            text_ctx = text_ctx[1:]
        else:
            text_ctx = []
        text_features = self.text_encoder(prompt_vectors, tokenized_prompts, text_ctx)
        return text_features, vis_ctx

    def forward(self, image):
        classnames = self.classnames
        classnames = [name.replace("_", " ") for name in classnames]
        prompts_ = classnames
        # print(f"Prompts: {prompts_}")
        prompts_ = torch.cat([clip.tokenize(p) for p in prompts_])
        prompts_ = prompts_.cuda()

        with torch.no_grad():
            text_features_ = self.clip_model_.encode_text(prompts_)
            text_features_ = text_features_ / text_features_.norm(dim=-1, keepdim=True)

        text_features, vis_ctx = self.encode_text(classnames, text_features_)
        image_features = self.encode_image(image, vis_ctx)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


class FedTPG(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.FedTPG.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.FedTPG.PREC == "fp32" or cfg.TRAINER.FedTPG.PREC == "amp":
            clip_model.float()

        print("Building custom CLIP")