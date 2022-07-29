<h2 align="center">🚧 Under Development 🚧</h2>

## Overview

**ViTX** is a **Vi**sion-**T**e**X**t representation learning framework on top of [PyTorch Lightning](https://www.pytorchlightning.ai/) and HuggingFace [Transformers](https://huggingface.co/) for Cross-Modal Perception research. It is designed to be readable and easily extensible, to allow users to quickly run and experiment with their own ideas.

### Currently Supported Methods

- [CLIP, 2021](https://arxiv.org/abs/2103.00020)
- [~~SLIP, 2021~~](https://arxiv.org/abs/2112.12750)
- [~~CyCLIP, 2022~~](https://arxiv.org/abs/2205.14459)
- [SimCLR, 2020](https://arxiv.org/abs/2002.05709)
- [~~BYOL, 2020~~](2006.07733)

## Installation

### Dependencies

ViTX requires **Python 3.6+**.

- hydra-core>=1.0.0
- numpy>=1.22.4
- rich>=12.4.4
- pytorch>=1.12.0
- torchvision>=0.13.0
- pytorch-lightning
- transformers
- wandb>=0.12.19

<!-- ### PyPI Installation
You can install Lightly and its dependencies from PyPI with:
```
pip install clhive
``` -->

### Manual Installation
It is strongly recommend that you install ViTX in a dedicated virtualenv, to avoid conflicting with your system packages.

```
git clone https://github.com/naderAsadi/ViTX.git
cd ViTX
pip install -e .
```


## How To Use

With `vitx` you can use uni-modal and multi-modal self-supervised methods in a modular way using the full power of PyTorch. Experiment with different backbones, models and loss functions. The framework has been designed to be easy to use from the ground up.

### Quick Start

```python

```


<details>
  <summary>More training examples</summary>
  
Train CLIP with ViT-base on COCO Captions dataset:

```
python main.py data=coco model/vision_model=vit-b  model/text_model=vit-b
```
  
</details>

## Reading The Commits
Here is a reference to what each emoji in the commits means:

* 📎 : Some basic updates.
* ♻️ : Refactoring.
* 💩 : Bad code, needs to be revised!
* 🐛 : Bug fix.
* 💡 : New feature.
* ⚡ : Performance Improvement.
