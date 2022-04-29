import copy
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch.nn as nn

from ..config import ModelConfig
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


def get_model(model_name: str, *args, **kwargs):

    assert model_name in MODEL_REGISTRY, "unknown model"
    model = MODEL_REGISTRY[model_name](*args, **kwargs)

    return model


# automatically import any Python files in the models/ directory
import_all_modules(FILE_ROOT, "vision_text.models")


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
from .dummy_dataclasses import VisionOutput, TextOutput, VisionTextOutput
