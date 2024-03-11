from collections import OrderedDict
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from clip import load, tokenize
from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from .custom_clip import TextEncoder
from data.imagnet_prompts import imagenet_classes
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets

_tokenizer = _Tokenizer()

DOWNLOAD_ROOT='~/.cache/clip'

class CoCoOpPromptLearner(nn.Module):
    def __init__(self, clip_model, classnames, n_ctx=4, ctx_init="a_photo_of_a", ctx_position='end'):
        super().__init__()
        n_cls = len(classnames)
        dtype = clip_model.dtype
        self.dtype = dtype
        self.device = clip_model.visual.conv1.weight.device
        ctx_dim = clip_model.ln_final.weight.shape[0]
        embed_dim = clip_model.text_projection.shape[1]
        self.ctx_dim = ctx_dim

        if ctx_init:
            # use given words to initialize context vectors
            print("Initializing the contect with given words: [{}]".format(ctx_init))
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = tokenize(ctx_init).to(self.device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            print("Random initialization: initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")
        self.prompt_prefix = prompt_prefix

        self.ctx = nn.Parameter(ctx_vectors) # to be optimized
        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(embed_dim, embed_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(embed_dim // 16, ctx_dim))
        ]))

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.ctx_init = ctx_init
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = ctx_position
        self.n_cls = n_cls
        self.n_ctx = n_ctx

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,     # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def reset_classnames(self, classnames, arch):
        self.n_cls = len(classnames)
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)

        clip, _, _ = load(arch, device=self.device, download_root=DOWNLOAD_ROOT)

        with torch.no_grad():
            embedding = clip.token_embedding(tokenized_prompts).type(self.dtype)

        self.token_prefix = embedding[:, :1, :]
        self.token_suffix = embedding[:, 1 + self.n_ctx :, :]  # CLS, EOS

        self.name_lens = name_lens
        self.tokenized_prompts = tokenized_prompts

    def forward(self, im_features, ctx_only=False):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx                     # (n_ctx, ctx_dim)
        bias = self.meta_net(im_features)  # (batch, ctx_dim)
        bias = bias.unsqueeze(1)           # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)             # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias           # (batch, n_ctx, ctx_dim)
        if ctx_only:
            return ctx_shifted # don't expand to n_cls, optimize one ctx for all classes
        
        # Use instance-conditioned context tokens for all classes
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)
        
        return prompts

class CoCoOpCLIP(nn.Module):
    def __init__(self, device, classnames, criterion='cosine', arch="ViT-L/14",
                        n_ctx=16, ctx_init="a_photo_of_a", ctx_position='end'):
        super().__init__()
        clip, _, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.image_encoder = clip.visual
        self.text_encoder = TextEncoder(clip)
        self.logit_scale = clip.logit_scale.data
        # prompt tuning
        self.prompt_generator = CoCoOpPromptLearner(clip, classnames, n_ctx, ctx_init, ctx_position)
        self.tokenized_prompts = self.prompt_generator.tokenized_prompts
        self.criterion = criterion
        self.dtype = clip.dtype

    def inference(self, image, label=None):
        tokenized_prompts = self.prompt_generator.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts = self.prompt_generator(image_features)
    
        logits = []
        for pts_i, imf_i in zip(prompts, image_features):
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            l_i = logit_scale * imf_i @ text_features.t()
            logits.append(l_i)
        logits = torch.stack(logits)

        return logits

    def gen_ctx(self, image, aug=False):
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                image_features = self.image_encoder(image.type(self.dtype))
                if aug:
                    image_feature_avg = image_features[0].unsqueeze(0)
                else:
                    image_feature_avg = image_features.mean(dim=0, keepdim=True)
                ctx = self.prompt_generator(image_feature_avg, ctx_only=True)

        return image_features, ctx.detach().clone()

    def forward_ctx(self, image_features, ctx):
        N = 1
        
        prefix = self.prompt_generator.token_prefix.expand(N, -1, -1, -1) # [N, n_cls, 1, dim]
        suffix = self.prompt_generator.token_suffix.expand(N, -1, -1, -1)
        # expand `ctx` n_cls times
        ctx = ctx.expand(self.prompt_generator.n_cls, -1, -1, -1)
        ctx = ctx.permute(1, 0, 2, 3)
        # ctx = ctx.reshape(N, self.prompt_generator.n_cls, -1, self.prompt_generator.ctx_dim)

        prompts = torch.cat([
            prefix,
            ctx,
            suffix
        ], dim=-2)

        # full_n_ctx = prompts.size()[-2]

        prompts = prompts.reshape(N * self.prompt_generator.n_cls, -1, self.prompt_generator.ctx_dim)
        tokenized_prompts = self.prompt_generator.tokenized_prompts
        tokenized_prompts = tokenized_prompts.repeat(N, 1)
        text_features = self.text_encoder(prompts, tokenized_prompts)

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        text_features = text_features.reshape(N, -1, image_features.size()[-1])

        logit_scale = self.logit_scale.exp()

        text_features = text_features.squeeze(0)
        logits = logit_scale * image_features @ text_features.t()

        return logits

    def forward(self, input):
        if isinstance(input, Tuple):
            image_features, ctx = input
            return self.forward_ctx(image_features, ctx)
        else:
            return self.inference(input)

def get_cocoop(clip_arch, test_set, device, n_ctx):
    if test_set in fewshot_datasets:
        classnames = eval("{}_classes".format(test_set.lower()))
    else:
        classnames = imagenet_classes
    
    model = CoCoOpCLIP(device, classnames, arch=clip_arch, n_ctx=n_ctx)

    return model