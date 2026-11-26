import os.path as osp
import os
import csv
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX, TrainerXU
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from tqdm import tqdm

_tokenizer = _Tokenizer()

import math
import numpy as np


def _required_file(path_value, config_key):
    """Resolve an external asset path and fail with an actionable message."""
    if not path_value:
        raise ValueError(
            f"{config_key} is not set. LUNA does not bundle teacher weights; "
            "provide the external asset path in the config or command line."
        )

    path = osp.realpath(osp.expanduser(os.path.expandvars(path_value)))
    if not osp.isfile(path):
        raise FileNotFoundError(f'{config_key} does not point to a file: "{path}"')

    return path


def _required_directory(path_value, config_key):
    """Resolve a required external directory without modifying it."""
    if not path_value:
        raise ValueError(
            f"{config_key} is not set. LUNA does not bundle teacher weights; "
            "provide the external checkpoint root in the config or command line."
        )

    path = osp.realpath(osp.expanduser(os.path.expandvars(path_value)))
    if not osp.isdir(path):
        raise FileNotFoundError(
            f'{config_key} does not point to a directory: "{path}"'
        )

    return path

def kd_loss(logits_student, logits_teacher, temperature, reduce=True):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    if reduce:
        loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    else:
        loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1)
    loss_kd *= temperature**2
    return loss_kd

def cc_loss(logits_student, logits_teacher, temperature, reduce=True):
    batch_size, class_num = logits_teacher.shape
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    student_matrix = torch.mm(pred_student.transpose(1, 0), pred_student)
    teacher_matrix = torch.mm(pred_teacher.transpose(1, 0), pred_teacher)
    if reduce:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2).sum() / class_num
    else:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2) / class_num
    return consistency_loss


def bc_loss(logits_student, logits_teacher, temperature, reduce=True):
    batch_size, class_num = logits_teacher.shape
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    student_matrix = torch.mm(pred_student, pred_student.transpose(1, 0))
    teacher_matrix = torch.mm(pred_teacher, pred_teacher.transpose(1, 0))
    if reduce:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2).sum() / batch_size
    else:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2) / batch_size
    return consistency_loss

def load_clip_to_cpu_teacher(cfg, zero_shot_model=False):
    backbone_name = cfg.TRAINER.LUNA.TEACHER_NAME
    model_path = _required_file(
        cfg.TRAINER.LUNA.TEACHER_CLIP_PATH,
        "TRAINER.LUNA.TEACHER_CLIP_PATH",
    )
    
    print(f"CLIP Teacher name is {backbone_name}")
    
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    # Build the externally prompt-tuned teacher architecture.
    design_details = {"trainer": 'IVLP',
                        "vision_depth": 9,
                        "language_depth": 9,
                        "vision_ctx": 4,
                        "language_ctx": 4}
    
    # design_details = {"trainer": 'CoOp',
    #                   "vision_depth": 0,
    #                   "language_depth": 0, "vision_ctx": 0,
    #                   "language_ctx": 0}
    
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'IVLP',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class VLPromptLearner_tea(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        # Make sure Language depth >= 1
        assert cfg.TRAINER.LUNA.PROMPT_DEPTH_TEXT >= 1, "In Independent VL prompting, Language prompt depth should be >=1" \
                                                        "\nPlease use VPT trainer if you want to learn only vision " \
                                                        "branch"
        n_ctx = cfg.TRAINER.LUNA.N_CTX_TEXT
        ctx_init = cfg.TRAINER.LUNA.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        
        self.trainer_name = cfg.TRAINER.NAME
        self.train_modal = cfg.TRAINER.MODAL
        
        if ctx_init and n_ctx <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print(f"Independent V-L design")
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of context words (tokens) for Language prompting: {n_ctx}")
        print(f"Number of context words (tokens) for Vision prompting: {cfg.TRAINER.LUNA.N_CTX_VISION}")
        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        
        print(f'classnames size is {len(classnames)}')

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        # self.name_lens = name_lens

        if self.train_modal == "base2novel":
            self.register_buffer("token_prefix", embedding[:math.ceil(self.n_cls / 2), :1, :])  # SOS
            self.register_buffer("token_suffix", embedding[:math.ceil(self.n_cls / 2), 1 + n_ctx:, :])  # CLS, EOS

            self.register_buffer("token_prefix2", embedding[math.ceil(self.n_cls / 2):, :1, :])  # SOS
            self.register_buffer("token_suffix2", embedding[math.ceil(self.n_cls / 2):, 1 + n_ctx:, :])  # CLS, EOS
            
        elif self.train_modal == "cross":
            self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
            
            self.register_buffer("token_prefix2", embedding[:, :1, :])  # SOS
            self.register_buffer("token_suffix2", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        # print(f'label is {label}')
        # if label is not None:
        #     prefix = prefix[label]
        #     suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        # print(f'ctx size is {ctx.size()}')

        prefix = self.token_prefix
        # print(f'prefix size is {prefix.size()}')
        
        suffix = self.token_suffix
        # print(f'suffix size is {suffix.size()}')

        if self.trainer_name == "LUNA" and self.train_modal == "base2novel":
            # print(f'n_cls is {self.n_cls}')
            prefix = torch.cat([prefix, self.token_prefix2], dim=0)
            suffix = torch.cat([suffix, self.token_suffix2], dim=0)

        prompts = self.construct_prompts(ctx, prefix, suffix)

        return prompts

class VLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        self.n_cls = n_cls
        # Make sure Language depth >= 1
        assert cfg.TRAINER.IVLP.PROMPT_DEPTH_TEXT >= 1, "In Independent VL prompting, Language prompt depth should be >=1" \
                                                        "\nPlease use VPT trainer if you want to learn only vision " \
                                                        "branch  "
        # n_ctx = cfg.TRAINER.IVLP.N_CTX_TEXT
        n_ctx = 4
        ctx_init = cfg.TRAINER.IVLP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        
        self.trainer_name = cfg.TRAINER.NAME
        self.train_modal = cfg.TRAINER.MODAL

        ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
        # ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)
        
        print(f"Independent V-L design")
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of context words (tokens) for Language prompting: {n_ctx}")
        print(f"Number of context words (tokens) for Vision prompting: {cfg.TRAINER.IVLP.N_CTX_VISION}")
        
        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        if self.train_modal == "base2novel":
            self.register_buffer("token_prefix", embedding[:math.ceil(self.n_cls / 2), :1, :])  # SOS
            self.register_buffer("token_suffix", embedding[:math.ceil(self.n_cls / 2), 1 + n_ctx:, :])  # CLS, EOS

            self.register_buffer("token_prefix2", embedding[math.ceil(self.n_cls / 2):, :1, :])  # SOS
            self.register_buffer("token_suffix2", embedding[math.ceil(self.n_cls / 2):, 1 + n_ctx:, :])  # CLS, EOS
            
        elif self.train_modal == "cross":
            self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
            
            self.register_buffer("token_prefix2", embedding[:, :1, :])  # SOS
            self.register_buffer("token_suffix2", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim
        
        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        
        if self.trainer_name == "LUNA" and self.train_modal == "base2novel":
            # print(f'n_cls is {self.n_cls}')
            prefix = torch.cat([prefix, self.token_prefix2], dim=0)
            suffix = torch.cat([suffix, self.token_suffix2], dim=0)
        
        prompts = self.construct_prompts(ctx, prefix, suffix)

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = VLPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.n_cls = len(classnames)

    def forward(self, image, label):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()



        prompts = self.prompt_learner().half()
        text_features = self.text_encoder(prompts, tokenized_prompts)
        image_features = self.image_encoder(image.type(self.dtype))

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # logits = logit_scale * image_features @ text_features.t()

        return logit_scale, image_features, text_features
    
class CustomCLIP_teacher(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = VLPromptLearner_tea(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model).cuda()
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.n_base_cls = math.ceil(len(classnames) / 2)
  
    def forward(self, image=None, label=None):
        
        prompts = self.prompt_learner()
        # Compute the prompted image and text features
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts.cuda(), tokenized_prompts.cuda())
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        logit_scale = self.logit_scale.exp()
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # Compute the prompted logits
        logits = logit_scale * image_features @ text_features.t()

        
        return logits


@TRAINER_REGISTRY.register()
class LUNA(TrainerXU):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.IVLP.PREC in ["fp16", "fp32", "amp"]

    def _resolve_teacher_checkpoint(self):
        cfg = self.cfg
        if cfg.TRAINER.MODAL == "base2novel":
            checkpoint_root = _required_directory(
                cfg.TRAINER.LUNA.TEACHER_CHECKPOINT_ROOT,
                "TRAINER.LUNA.TEACHER_CHECKPOINT_ROOT",
            )
            model_path = osp.join(
                checkpoint_root,
                str(cfg.DATASET.NAME),
                "VLPromptLearner",
                "model-best.pth.tar",
            )
            model_path = _required_file(
                model_path,
                "LUNA base-to-new teacher checkpoint",
            )
        elif cfg.TRAINER.MODAL == "cross":
            model_path = _required_file(
                cfg.TRAINER.LUNA.CROSS_DATASET_TEACHER_CHECKPOINT,
                "TRAINER.LUNA.CROSS_DATASET_TEACHER_CHECKPOINT",
            )
        else:
            raise ValueError(
                f'Unsupported TRAINER.MODAL="{cfg.TRAINER.MODAL}"; '
                'expected "base2novel" or "cross".'
            )
        return model_path

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        self.n_cls = len(classnames)

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model_teacher = load_clip_to_cpu_teacher(cfg)

        if cfg.TRAINER.IVLP.PREC == "fp32" or cfg.TRAINER.IVLP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)
        self.model_teacher = CustomCLIP_teacher(cfg, classnames, clip_model_teacher)
        
        model_path = self._resolve_teacher_checkpoint()

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"
        
        self.train_modal = cfg.TRAINER.MODAL
        
        checkpoint = load_checkpoint(model_path)
        state_dict = checkpoint["state_dict"]
        
        if "prompt_learner.token_prefix" in state_dict:
            del state_dict["prompt_learner.token_prefix"]
        if "prompt_learner.token_prefix2" in state_dict:
            del state_dict["prompt_learner.token_prefix2"]

        if "prompt_learner.token_suffix" in state_dict:
            del state_dict["prompt_learner.token_suffix"]
        if "prompt_learner.token_suffix2" in state_dict:
            del state_dict["prompt_learner.token_suffix2"]
        
        self.model_teacher.load_state_dict(state_dict, strict=False)
        self.model_teacher.to(self.device)
        self.model_teacher.eval()

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("VLPromptLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.IVLP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        # device_count = torch.cuda.device_count()
        # if device_count > 1:
        #     print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        #     self.model = nn.DataParallel(self.model)
        
        self.temperature = 1.0
        self.kd_loss_weight = 1.0

    def forward_backward(self, batch_x, batch_u):
        # image, label = self.parse_batch_train(batch)
        image_weak, image_strong, label = self.parse_batch_train(batch_x, batch_u)
        

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.IVLP.PREC
        if prec == "amp":
            with autocast():
                loss = model(image_weak, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            logit_scale, img_ft_weak, text_features = model(image_weak, label)

            logits_student_weak = logit_scale * img_ft_weak @ text_features.t()
            
            logit_scale, img_ft_strong, text_features = model(image_strong, label)
            logits_student_strong = logit_scale * img_ft_strong @ text_features.t()
            
            with torch.no_grad():
                logits_teacher_weak = self.model_teacher(image_weak, label)
                logits_teacher_strong = self.model_teacher(image_strong, label)
            
            batch_size, class_num = logits_student_strong.shape
            
            pred_teacher_weak = F.softmax(logits_teacher_weak.detach(), dim=1)
            confidence, pseudo_labels = pred_teacher_weak.max(dim=1)
            confidence = confidence.detach()
            conf_thresh = np.percentile(
                confidence.cpu().numpy().flatten(), 50
            )
            mask = confidence.le(conf_thresh).bool()
            
            class_confidence = torch.sum(pred_teacher_weak, dim=0)
            class_confidence = class_confidence.detach()
            class_confidence_thresh = np.percentile(
                class_confidence.cpu().numpy().flatten(), 50
            )
            class_conf_mask = class_confidence.le(class_confidence_thresh).bool()

            loss_kd_weak = self.kd_loss_weight * ((kd_loss(
            logits_student_weak,
            logits_teacher_weak,
            self.temperature,
            # reduce=False
            ) * mask).mean()) + self.kd_loss_weight * ((kd_loss(
                logits_student_weak,
                logits_teacher_weak,
                3.0,
                # reduce=False
            ) * mask).mean()) + self.kd_loss_weight * ((kd_loss(
                logits_student_weak,
                logits_teacher_weak,
                5.0,
                # reduce=False
            ) * mask).mean()) + self.kd_loss_weight * ((kd_loss(
                logits_student_weak,
                logits_teacher_weak,
                2.0,
                # reduce=False
            ) * mask).mean()) + self.kd_loss_weight * ((kd_loss(
                logits_student_weak,
                logits_teacher_weak,
                6.0,
                # reduce=False
            ) * mask).mean())

            loss_kd_strong = self.kd_loss_weight * kd_loss(
                logits_student_strong,
                logits_teacher_strong,
                self.temperature,
            ) + self.kd_loss_weight * kd_loss(
                logits_student_strong,
                logits_teacher_strong,
                3.0,
            ) + self.kd_loss_weight * kd_loss(
                logits_student_strong,
                logits_teacher_strong,
                5.0,
            ) + self.kd_loss_weight * kd_loss(
                logits_student_weak,
                logits_teacher_weak,
                2.0,
            ) + self.kd_loss_weight * kd_loss(
                logits_student_weak,
                logits_teacher_weak,
                6.0,
            )

            loss_cc_weak = self.kd_loss_weight * ((cc_loss(
                logits_student_weak,
                logits_teacher_weak,
                self.temperature,
                # reduce=False
            ) * class_conf_mask).mean()) + self.kd_loss_weight * ((cc_loss(
                logits_student_weak,
                logits_teacher_weak,
                3.0,
            ) * class_conf_mask).mean()) + self.kd_loss_weight * ((cc_loss(
                logits_student_weak,
                logits_teacher_weak,
                5.0,
            ) * class_conf_mask).mean()) + self.kd_loss_weight * ((cc_loss(
                logits_student_weak,
                logits_teacher_weak,
                2.0,
            ) * class_conf_mask).mean()) + self.kd_loss_weight * ((cc_loss(
                logits_student_weak,
                logits_teacher_weak,
                6.0,
            ) * class_conf_mask).mean())
            
            loss_cc_strong = self.kd_loss_weight * cc_loss(
                logits_student_strong,
                logits_teacher_strong,
                self.temperature,
            ) + self.kd_loss_weight * cc_loss(
                logits_student_strong,
                logits_teacher_strong,
                3.0,
            ) + self.kd_loss_weight * cc_loss(
                logits_student_strong,
                logits_teacher_strong,
                5.0,
            ) + self.kd_loss_weight * cc_loss(
                logits_student_weak,
                logits_teacher_weak,
                2.0,
            ) + self.kd_loss_weight * cc_loss(
                logits_student_weak,
                logits_teacher_weak,
                6.0,
            )

            loss_bc_weak = self.kd_loss_weight * ((bc_loss(
                logits_student_weak,
                logits_teacher_weak,
                self.temperature,
            ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
                logits_student_weak,
                logits_teacher_weak,
                3.0,
            ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
                logits_student_weak,
                logits_teacher_weak,
                5.0,
            ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
                logits_student_weak,
                logits_teacher_weak,
                2.0,
            ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
                logits_student_weak,
                logits_teacher_weak,
                6.0,
            ) * mask).mean())
            
            loss_bc_strong = self.kd_loss_weight * ((bc_loss(
                logits_student_strong,
                logits_teacher_strong,
                self.temperature,
            ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
                logits_student_strong,
                logits_teacher_strong,
                3.0,
            ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
                logits_student_strong,
                logits_teacher_strong,
                5.0,
            ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
                logits_student_strong,
                logits_teacher_strong,
                2.0,
            ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
                logits_student_strong,
                logits_teacher_strong,
                6.0,
            ) * mask).mean())

            
            # losses_dict = {
            # "loss_kd": loss_kd_weak + loss_kd_strong,
            # "loss_cc": loss_cc_weak + loss_cc_strong,
            # "loss_bc": loss_bc_weak + loss_bc_strong
            # }
            
            losses_dict = {
            "loss_kd": loss_kd_weak,
            "loss_cc": loss_cc_weak,
            "loss_bc": loss_bc_weak
            }
            
            
            loss = sum([l.mean() for l in losses_dict.values()])
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            # def closure():
            #     optim.zero_grad()
            #     loss = L_ukd
            #     loss.backward(retain_graph=True)
            #     return loss

            # loss = optim.step(closure)

        loss_summary = {"loss": loss.item()}

        # Record the three distillation terms exactly as they participate in
        # the optimization above. Values are averaged over all batches in an
        # epoch and written once per epoch for downstream curve plotting.
        if prec != "amp":
            component_values = {
                "loss_kd": loss_kd_weak.item(),
                "loss_cc": loss_cc_weak.item(),
                "loss_bc": loss_bc_weak.item(),
            }
            loss_summary.update(component_values)

            if self.batch_idx == 0:
                self._epoch_loss_sums = {
                    "loss_kd": 0.0,
                    "loss_cc": 0.0,
                    "loss_bc": 0.0,
                }
                self._epoch_loss_count = 0

            for name, value in component_values.items():
                self._epoch_loss_sums[name] += value
            self._epoch_loss_count += 1

            if (self.batch_idx + 1) == self.num_batches:
                os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
                csv_path = osp.join(self.cfg.OUTPUT_DIR, "loss_trends.csv")
                file_mode = "w" if self.epoch == 0 else "a"
                fieldnames = [
                    "epoch",
                    "loss_kd",
                    "loss_cc",
                    "loss_bc",
                ]
                row = {
                    "epoch": self.epoch + 1,
                    "loss_kd": self._epoch_loss_sums["loss_kd"] / self._epoch_loss_count,
                    "loss_cc": self._epoch_loss_sums["loss_cc"] / self._epoch_loss_count,
                    "loss_bc": self._epoch_loss_sums["loss_bc"] / self._epoch_loss_count,
                }
                with open(csv_path, file_mode, newline="") as csv_file:
                    writer = csv.DictWriter(
                        csv_file,
                        fieldnames=fieldnames,
                        lineterminator="\n",
                    )
                    if file_mode == "w":
                        writer.writeheader()
                    writer.writerow(row)
                print(f"Epoch loss averages saved to {csv_path}: {row}")

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    # def parse_batch_train(self, batch):
    #     input = batch["img"]
    #     label = batch["label"]
    #     input = input.to(self.device)
    #     label = label.to(self.device)
    #     return input, label
    
    def parse_batch_train(self, batch_x, batch_u):
        image_weak = batch_x["img"]
        label = batch_x["label"]
        image_strong = batch_u["img"]
        image_weak = image_weak.to(self.device)
        label = label.to(self.device)
        image_strong = image_strong.to(self.device)
        return image_weak, image_strong, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
            
    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()


        if split is None:
            split = self.cfg.TEST.SPLIT
            
        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        elif split == "train":
            data_loader = self.train_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")
        
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            image, label = self.parse_batch_test(batch)
            
            # with torch.no_grad():
            #     tea_image_features, tea_text_features, tea_logits = self.model_teacher(image, label)
            
            logit_scale, image_ft, text_features = self.model(image, label)
            
            if self.train_modal == "base2novel":
                if split == "val":
                    output = logit_scale * image_ft @ text_features[:math.ceil(self.n_cls / 2),:].t()
                elif split == "test":
                    output = logit_scale * image_ft @ text_features[math.ceil(self.n_cls / 2):,:].t()
            elif self.train_modal == "cross":
                output = logit_scale * image_ft @ text_features.t()
            
            self.evaluator.process(output, label) 

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)
            
        return list(results.values())[0]
    
    
