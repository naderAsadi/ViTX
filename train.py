from PIL import Image
import random
import requests

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.datasets as datasets

from transformers import (
    VisionTextDualEncoderModel,
    VisionTextDualEncoderProcessor,
    ViTFeatureExtractor,
    BertTokenizer,
)
from transformers import (
    CLIPVisionModel,
    CLIPTextModel,
    CLIPFeatureExtractor,
    CLIPTokenizer,
    CLIPProcessor
)

from vision_text.models import VisionTextModel
from vision_text.config import config_parser


def collate_fn(batch):
    return tuple(zip(*batch))

def train():
    config = config_parser(config_path='./configs/', config_name="default", job_name="test")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
    # processor = VisionTextDualEncoderProcessor(feature_extractor, tokenizer)

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    # feature_extractor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

    model = VisionTextModel(model_config=config.model, vision_model=vision_model, text_model=text_model)
    # model = VisionTextModel(model_config=config.model)
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=5e-3, momentum=0.9, weight_decay=1e-4)

    # loading data
    train_data_path = "../datasets/coco-caption/images/train2014/"
    train_ann_path = "../datasets/coco-caption/annotations/captions_train2014.json"

    transform_train = T.Compose(
        [
            T.Resize(224),
            T.RandomCrop((224, 224), padding=4, fill=-1),
            # T.RandomHorizontalFlip(),
            # T.ColorJitter(0.4, 0.4, 0.4),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    coco_train_dataset = datasets.CocoCaptions(
        root=train_data_path, annFile=train_ann_path, transform=transform_train
    )
    train_loader = DataLoader(
        coco_train_dataset,
        batch_size=32,
        num_workers=8,
        shuffle=True,
        collate_fn=collate_fn,
    )

    model.train()
    n_epochs = 30
    for epoch in range(n_epochs):
        sum_loss = 0
        for i, (images, captions) in enumerate(train_loader):

            text = []
            for cap in captions:
                text.append(random.choice(cap))
        
            tokens = tokenizer(text)
            print(tokens.keys())

            inputs = processor(
                text=text, images=images, return_tensors="pt", padding=True
            )

            for key, value in inputs.items():
                inputs[key] = value.to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=inputs.input_ids,
                attention_mask=None,
                pixel_values=inputs.pixel_values,
                return_loss=True,
            )
            loss, logits_per_image = outputs.loss, outputs.logits_per_image  # this is the image-text similarity score
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            print(f"Epoch: {epoch}/{n_epochs} Iter: {i}/{len(train_loader)} Loss: {loss.item()}", end='\r') #Similarity: {logits_per_image.cpu().detach().numpy().reshape(-1)}"

        print(f"Avg Loss: {sum_loss / n_epochs}")

    # save and load from pretrained
    # model.save_pretrained("vit-bert")


def evaluate():
    model = VisionTextDualEncoderModel.from_pretrained("vit-bert")

    inference
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1) 


if __name__ == '__main__':
    train()