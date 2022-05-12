import copy
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch.nn as nn
from transformers import (
    CLIPVisionModel,
    CLIPTextModel,
    CLIPVisionConfig,
    CLIPTextConfig,
)

from ..config import VisionModelConfig, TextModelConfig
from ..utils.registry_utils import import_all_modules


FILE_ROOT = Path(__file__).parent

MODEL_REGISTRY = {}
MODEL_CLASS_NAMES = set()
MODEL_REGISTRY_TB = {}
MODEL_CLASS_NAMES_TB = {}


def register_model(name, bypass_checks=False):
    """Register a :class:`torch.nn.Module` subclass.
    This decorator allows instantiating a subclass of :class:`torch.nn.Module`
    from a configuration file. To use it, apply this decorator to a `torch.nn.Module`
    subclass.
    Args:
        name ([type]): [description]
        bypass_checks (bool, optional): [description]. Defaults to False.
    """

    def register_model_cls(cls):
        if not bypass_checks:
            if name in MODEL_REGISTRY:
                raise ValueError(
                    f"Cannot register duplicate model ({name}). Already registered at \n{MODEL_REGISTRY_TB[name]}\n"
                )
        tb = "".join(traceback.format_stack())
        MODEL_REGISTRY[name] = cls
        MODEL_CLASS_NAMES.add(cls.__name__)
        MODEL_REGISTRY_TB[name] = tb
        MODEL_CLASS_NAMES_TB[cls.__name__] = tb
        return cls

    return register_model_cls


def get_model(model_config: Union[VisionModelConfig, TextModelConfig], **kwargs):

    if model_config.name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_config.name](
            output_dim=model_config.embed_dim,
            pretrained=model_config.pretrained,
            **kwargs,
        )

    model = None
    if "tokenizer" not in model_config.keys():
        model = CLIPVisionModel.from_pretrained(model_config.name)
    else:
        if model_config.pretrained:
            model = CLIPTextModel.from_pretrained(model_config.name)
        else:
            model = CLIPTextModel(
                CLIPTextConfig(
                    vocab_size=model_config.vocab_size,
                    hidden_size=model_config.embed_dim,
                    num_hidden_layers=model_config.n_hidden_layers,
                    num_attention_heads=model_config.n_attention_heads,
                    max_position_embeddings=model_config.max_token_length,
                )
            )

    assert model is not None, f"unknown model: {model_config.name}"

    return model


# automatically import any Python files in the models/ directory
import_all_modules(FILE_ROOT, "vitx.models")


from .mlp import LinearClassifier, MLP
from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .vision_transformer import (
    Transformer,
    VisualTransformer,
    vit_tiny_patch16,
    vit_small_patch16,
    vit_small_patch32,
    vit_base_patch16,
    vit_base_patch32,
    vit_large_patch14,
    vit_large_patch16,
    vit_large_patch32,
    vit_huge_patch14,
    vit_giant_patch14,
)
from .vision_text_dual_encoder import VisionTextEncoder
from .dummy_dataclasses import ModelOutput, VisionTextDualOutput
