import traceback
from pathlib import Path

from ..config import DataConfig
from ..utils.registry_utils import import_all_modules


FILE_ROOT = Path(__file__).parent

DATASET_REGISTRY = {}
DATASET_REGISTRY_TB = {}
DATASET_CLASS_NAMES = set()
DATASET_CLASS_NAMES_TB = {}


def register_dataset(name, bypass_checks=False):
    """Registers a :class:`torch.utils.data.Dataset` subclass.
    This decorator allows to instantiate a subclass of Dataset
    from a configuration file, even if the class itself is not
    part of the framework. To use it, apply this decorator to a
    `torch.utils.data.Dataset` subclass like this:
    .. code-block:: python
      @register_dataset("my_dataset")
      class MyDataset(Dataset):
          ...
    """

    def register_dataset_cls(cls):
        if not bypass_checks:
            if name in DATASET_REGISTRY:
                msg = "Cannot register duplicate dataset ({}). Already registered at \n{}\n"
                raise ValueError(msg.format(name, DATASET_REGISTRY_TB[name]))
            if cls.__name__ in DATASET_CLASS_NAMES:
                raise ValueError(
                    f"Cannot register dataset with duplicate class name({cls.__name__}). Previously registered at \n{DATASET_CLASS_NAMES_TB[cls.__name__]}\n"
                )
        tb = "".join(traceback.format_stack())
        DATASET_REGISTRY[name] = cls
        DATASET_CLASS_NAMES.add(cls.__name__)
        DATASET_REGISTRY_TB[name] = tb
        DATASET_CLASS_NAMES_TB[cls.__name__] = tb
        return cls

    return register_dataset_cls


def get_dataset(data_config: DataConfig, split: str = "train"):

    dataset = DATASET_REGISTRY[data_config.dataset].from_config(
        data_config=data_config, split=split
    )
    return dataset


# automatically import any Python files in the data/ directory
import_all_modules(FILE_ROOT, "vision_text.data")

# from .image_text import ImageTextDataset
from .coco import COCODataset
from .mpi import MPIVideoDataset

from .helper import get_dataloaders
