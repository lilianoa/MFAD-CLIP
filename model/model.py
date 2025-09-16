﻿from typing import Union, List
from pkg_resources import packaging
import torch
import torch.nn as nn
from copy import deepcopy
import clip
from clip import load
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result

class PromptLearner(nn.Module):
    def __init__(self, clip_model, cfg, classnames=['object']):
        super(PromptLearner, self).__init__()
        n_cls = len(classnames)
        n_ctx = cfg['n_ctx']
        n_ctx_pos = n_ctx
        n_ctx_neg = n_ctx
        ctx_init_pos = cfg['ctx_init_pos']
        ctx_init_neg = cfg['ctx_init_neg']
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        n_prompts = cfg['n_prompts']
        self.n_prompts_pos = n_prompts
        self.n_prompts_neg = n_prompts
        self.normal_states = ["{}", ]
        self.anomaly_states = ["damaged {}", ]

        # When multiple normal/abnormal text prompts are required, repeat the template multiple times
        if self.n_prompts_pos:
            self.normal_states = self.normal_states * self.n_prompts_pos
        if self.n_prompts_neg:
            self.anomaly_states = self.anomaly_states * self.n_prompts_neg

        if ctx_init_pos and ctx_init_neg:
            # use given words to initialize context vectors
            ctx_init_pos = ctx_init_pos.replace("_", " ")
            n_ctx_pos = len(ctx_init_pos.split(" "))
            ctx_init_neg = ctx_init_neg.replace("_", " ")
            n_ctx_neg = len(ctx_init_neg.split(" "))
            prompt_pos = tokenize(ctx_init_pos)
            prompt_neg = tokenize(ctx_init_neg)
            with torch.no_grad():
                embedding_pos = clip_model.token_embedding(prompt_pos).type(dtype)
                embedding_neg = clip_model.token_embedding(prompt_neg).type(dtype)
            ctx_vectors_pos = embedding_pos[0, 1: 1 + n_ctx_pos, :]
            prompt_prefix_pos = ctx_init_pos
            ctx_vectors_neg = embedding_neg[0, 1: 1 + n_ctx_neg, :]
            prompt_prefix_neg = ctx_init_neg

        else:
            # random initialization
            if True:
                print("Initializing class-specific contexts")
                ctx_vectors_pos = torch.empty(n_cls, self.n_prompts_pos, n_ctx_pos, ctx_dim, dtype=dtype)
                ctx_vectors_neg = torch.empty(n_cls, self.n_prompts_neg, n_ctx_neg, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors_pos = torch.empty(n_ctx_pos, ctx_dim, dtype=dtype)
                ctx_vectors_neg = torch.empty(n_ctx_neg, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors_pos, std=0.02)
            nn.init.normal_(ctx_vectors_neg, std=0.02)
            prompt_prefix_pos = " ".join(["X"] * n_ctx_pos)
            prompt_prefix_neg = " ".join(["X"] * n_ctx_neg)

        print(f'Initial context_pos: "{prompt_prefix_pos}"')
        print(f'Initial context_neg: "{prompt_prefix_neg}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx_pos = nn.Parameter(ctx_vectors_pos)  # to be optimized (n_cls, n_prompts_pos, n_ctx_pos, ctx_dim)
        self.ctx_neg = nn.Parameter(ctx_vectors_neg)  # to be optimized (n_cls, n_prompts_neg, n_ctx_neg, ctx_dim)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts_pos = [prompt_prefix_pos + " " + template.format(name) + "." for template in self.normal_states for name in classnames]
        prompts_neg = [prompt_prefix_neg + " " + template.format(name) + "." for template in self.anomaly_states for name in classnames]

        tokenized_prompts_pos = torch.cat([tokenize(p) for p in prompts_pos])
        tokenized_prompts_neg = torch.cat([tokenize(p) for p in prompts_neg])
        with torch.no_grad():
            embedding_pos = clip_model.token_embedding(tokenized_prompts_pos).type(dtype)  # (n_cls * n_prompts_pos, n_ctx_pos, ctx_dim)
            embedding_neg = clip_model.token_embedding(tokenized_prompts_neg).type(dtype)  # (n_cls * n_prompts_neg, n_ctx_neg, ctx_dim)
            n, l, d = embedding_pos.shape
            embedding_pos = embedding_pos.reshape(-1, n_cls, l, d).permute(1, 0, 2, 3)  # (n_cls, n_prompts_pos, n_ctx_pos, ctx_dim)
            embedding_neg = embedding_neg.reshape(-1, n_cls, l, d).permute(1, 0, 2, 3)  # (n_cls, n_prompts_neg, n_ctx_neg, ctx_dim)
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix_pos", embedding_pos[:, :, :1, :])  # SOS
        self.register_buffer("token_suffix_pos", embedding_pos[:, :, 1 + n_ctx:, :])  # CLS, EOS
        self.register_buffer("token_prefix_neg", embedding_neg[:, :, :1, :])  # SOS
        self.register_buffer("token_suffix_neg", embedding_neg[:, :, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx_neg = n_ctx_neg
        self.n_ctx_pos = n_ctx_pos
        self.tokenized_prompts_pos = tokenized_prompts_pos  # torch.Tensor
        self.tokenized_prompts_neg = tokenized_prompts_neg  # torch.Tensor
        self.name_lens = name_lens

    def forward(self):
        ctx_pos = self.ctx_pos
        ctx_neg = self.ctx_neg

        prefix_pos = self.token_prefix_pos
        suffix_pos = self.token_suffix_pos
        prefix_neg = self.token_prefix_neg
        suffix_neg = self.token_suffix_neg

        prompts_pos = torch.cat(
            [
                prefix_pos,  # (n_cls, n_prompts, 1, dim)
                ctx_pos,  # (n_cls, n_prompts, n_ctx, dim)
                suffix_pos,  # (n_cls, n_prompts, *, dim)
            ],
            dim=2,
        )
        prompts_neg = torch.cat(
            [
                prefix_neg,  # (n_cls, n_prompts, 1, dim)
                ctx_neg,  # (n_cls, n_prompts, n_ctx, dim)
                suffix_neg,  # (n_cls, n_prompts, *, dim)
            ],
            dim=2,
        )

        _, _, l, d = prompts_pos.shape
        prompts_pos = prompts_pos.reshape(-1, l, d)  # (n_cls*n_prompts, l, dim)
        _, _, l, d = prompts_neg.shape
        prompts_neg = prompts_neg.reshape(-1, l, d)  # (n_cls*n_prompts, l, dim)
        prompts = torch.cat([prompts_pos, prompts_neg], dim=0)
        tokenized_prompts = torch.cat((self.tokenized_prompts_pos, self.tokenized_prompts_neg), dim=0)

        return prompts, tokenized_prompts


def _get_clones(module, N):
    return nn.ModuleList([deepcopy(module) for i in range(N)])

class PromptLearner(nn.Module):
    def __init__(self, clip_model, design_details, classnames=["object"]):
        super().__init__()

        self.n_cls = len(classnames)
        self.n_ctx = design_details["Prompt_length"]
        n_ctx_pos = self.n_ctx
        n_ctx_neg = self.n_ctx
        self.text_encoder_n_ctx = design_details["learnable_text_embedding_length"]
        ctx_init_pos = ""
        ctx_init_neg = ""
        dtype = clip_model.transformer.get_cast_dtype()

        ctx_dim = clip_model.ln_final.weight.shape[0]

        self.classnames = classnames

        self.state_normal_list = [
            "{}",
        ]

        self.state_anomaly_list = [
            "damaged {}",
        ]

        normal_num = len(self.state_normal_list)
        anormaly_num = len(self.state_anomaly_list)
        self.normal_num = normal_num
        self.anormaly_num = anormaly_num

        if ctx_init_pos and ctx_init_neg:
            # use given words to initialize context vectors
            ctx_init_pos = ctx_init_pos.replace("_", " ")
            ctx_init_neg = ctx_init_neg.replace("_", " ")
            n_ctx_pos = len(ctx_init_pos.split(" "))
            n_ctx_neg = len(ctx_init_neg.split(" "))

            prompt_pos = tokenize(ctx_init_pos)
            prompt_neg = tokenize(ctx_init_neg)
            with torch.no_grad():
                
                embedding_pos = clip_model.token_embedding(prompt_pos).type(dtype)
                embedding_neg = clip_model.token_embedding(prompt_neg).type(dtype)
            
            ctx_vectors_pos = embedding_pos[0, 1: 1 + n_ctx_pos, :]
            ctx_vectors_neg = embedding_neg[0, 1: 1 + n_ctx_neg, :]
            prompt_prefix_pos = ctx_init_pos
            prompt_prefix_neg = ctx_init_neg
            if True:
                ctx_vectors_pos_ = []
                ctx_vectors_neg_ = []
                for _ in range(self.n_cls):
                    ctx_vectors_pos_.append(deepcopy(ctx_vectors_pos))
                    ctx_vectors_neg_.append(deepcopy(ctx_vectors_neg))
                ctx_vectors_pos = torch.stack(ctx_vectors_pos_, dim=0)
                ctx_vectors_neg = torch.stack(ctx_vectors_neg_, dim=0)

        else:
            # Random Initialization
                print("Initializing class-specific contexts")
                ctx_vectors_pos = torch.empty(self.n_cls, self.normal_num, n_ctx_pos, ctx_dim, dtype=dtype)
                ctx_vectors_neg = torch.empty(self.n_cls, self.anormaly_num, n_ctx_neg, ctx_dim, dtype=dtype)

            nn.init.normal_(ctx_vectors_pos, std=0.02)
            nn.init.normal_(ctx_vectors_neg, std=0.02)
            prompt_prefix_pos = " ".join(["X"] * n_ctx_pos)
            prompt_prefix_neg = " ".join(["X"] * n_ctx_neg)
        self.compound_prompts_depth = design_details["learnable_text_embedding_depth"]
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(self.text_encoder_n_ctx, ctx_dim))
                                                       for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            print("single_para", single_para.shape)
            nn.init.normal_(single_para, std=0.02)

        single_layer = nn.Linear(ctx_dim, 896)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)

        self.ctx_pos = nn.Parameter(ctx_vectors_pos)  # to be optimized
        self.ctx_neg = nn.Parameter(ctx_vectors_neg)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]

        prompts_pos = [prompt_prefix_pos + " " + template.format(name) + "." for template in self.state_normal_list for
                       name in classnames]
        prompts_neg = [prompt_prefix_neg + " " + template.format(name) + "." for template in self.state_anomaly_list for
                       name in classnames]

        tokenized_prompts_pos = []
        tokenized_prompts_neg = []

        for p_pos in prompts_pos:
            tokenized_prompts_pos.append(tokenize(p_pos))
        for p_neg in prompts_neg:
            tokenized_prompts_neg.append(tokenize(p_neg))
        tokenized_prompts_pos = torch.cat(tokenized_prompts_pos)
        tokenized_prompts_neg = torch.cat(tokenized_prompts_neg)
        
        with torch.no_grad():
            embedding_pos = clip_model.token_embedding(tokenized_prompts_pos).type(dtype)
            embedding_neg = clip_model.token_embedding(tokenized_prompts_neg).type(dtype)
            n, l, d = embedding_pos.shape
            print("embedding_pos", embedding_pos.shape)
            embedding_pos = embedding_pos.reshape(normal_num, self.n_cls, l, d).permute(1, 0, 2, 3)
            embedding_neg = embedding_neg.reshape(anormaly_num, self.n_cls, l, d).permute(1, 0, 2, 3)

        self.register_buffer("token_prefix_pos", embedding_pos[:, :, :1, :])
        self.register_buffer("token_suffix_pos", embedding_pos[:, :, 1 + n_ctx_pos:, :])
        self.register_buffer("token_prefix_neg", embedding_neg[:, :, :1, :])
        self.register_buffer("token_suffix_neg", embedding_neg[:, :, 1 + n_ctx_neg:, :])

        n, d = tokenized_prompts_pos.shape
        tokenized_prompts_pos = tokenized_prompts_pos.reshape(normal_num, self.n_cls, d).permute(1, 0, 2)

        n, d = tokenized_prompts_neg.shape
        tokenized_prompts_neg = tokenized_prompts_neg.reshape(anormaly_num, self.n_cls, d).permute(1, 0, 2)

        self.n_ctx_pos = n_ctx_pos
        self.n_ctx_neg = n_ctx_neg
        # tokenized_prompts = torch.cat([tokenized_prompts_pos, tokenized_prompts_neg], dim=0)  # torch.Tensor
        self.register_buffer("tokenized_prompts_pos", tokenized_prompts_pos)
        self.register_buffer("tokenized_prompts_neg", tokenized_prompts_neg)
        print("tokenized_prompts shape", self.tokenized_prompts_pos.shape, self.tokenized_prompts_neg.shape)

    def forward(self, cls_id=None):

        ctx_pos = self.ctx_pos
        ctx_neg = self.ctx_neg
        ctx_pos = self.ctx_pos
        ctx_neg = self.ctx_neg
        # print("shape", self.ctx_pos[0:1].shape, ctx_pos.shape)
        prefix_pos = self.token_prefix_pos
        prefix_neg = self.token_prefix_neg
        suffix_pos = self.token_suffix_pos
        suffix_neg = self.token_suffix_neg

        # print(prefix_pos.shape, prefix_neg.shape)

        prompts_pos = torch.cat(
            [
                # N(the number of template), 1, dim
                prefix_pos,  # (n_cls, 1, dim)
                ctx_pos,  # (n_cls, n_ctx, dim)
                suffix_pos,  # (n_cls, *, dim)
            ],
            dim=2,
        )

        prompts_neg = torch.cat(
            [
                prefix_neg,  # (n_cls, 1, dim)
                ctx_neg,  # (n_cls, n_ctx, dim)
                suffix_neg,  # (n_cls, *, dim)
            ],
            dim=2,
        )
        _, _, l, d = prompts_pos.shape
        prompts_pos = prompts_pos.reshape(-1, l, d)
        _, _, l, d = prompts_neg.shape
        prompts_neg = prompts_neg.reshape(-1, l, d)
        prompts = torch.cat([prompts_pos, prompts_neg], dim=0)

        _, l, d = self.tokenized_prompts_pos.shape
        tokenized_prompts_pos = self.tokenized_prompts_pos.reshape(-1, d)
        _, l, d = self.tokenized_prompts_neg.shape
        tokenized_prompts_neg = self.tokenized_prompts_neg.reshape(-1, d)
        tokenized_prompts = torch.cat((tokenized_prompts_pos, tokenized_prompts_neg), dim=0)

        return prompts, tokenized_prompts, self.compound_prompts_text
