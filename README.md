**Under Development!** 🚧

## Overview

**ViTX** is a **Vi**sion-**T**e**X**t representation learning framework on top of [PyTorch Lightning](https://www.pytorchlightning.ai/) and HuggingFace [Transformers](https://huggingface.co/) for Cross-Modal Perception research.

## How To Use

<details>
  <summary>Training examples</summary>
  
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
