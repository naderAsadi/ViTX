from PIL import Image
import requests
import torch
from transformers import (
    VisionTextDualEncoderModel,
    VisionTextDualEncoderProcessor,
    ViTFeatureExtractor,
    BertTokenizer,
)

from vision_text.models import VisionTextModel
from vision_text.config import config_parser


config = config_parser(config_path='./configs/', config_name="default", job_name="test")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
processor = VisionTextDualEncoderProcessor(feature_extractor, tokenizer)

model = VisionTextModel.from_vision_text_pretrained(
    vision_model_name_or_path="google/vit-base-patch16-224", 
    text_model_name_or_path="bert-base-uncased"
)
model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# contrastive training
urls = [
    "http://images.cocodataset.org/val2017/000000039769.jpg",
    "https://farm3.staticflickr.com/2674/5850229113_4fe05d5265_z.jpg",
]
images = [Image.open(requests.get(url, stream=True).raw) for url in urls]

for i in range(200):
    inputs = processor(
        text=["a photo of a cat", "a photo of a dog"], images=images, return_tensors="pt", padding=True
    )

    for key, value in inputs.items():
        inputs[key] = value.to(device)

    optimizer.zero_grad()
    outputs = model(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        pixel_values=inputs.pixel_values,
        return_loss=True,
    )
    loss, logits_per_image = outputs.loss, outputs.logits_per_image  # this is the image-text similarity score
    loss.backward()
    optimizer.step()

    # print(f"Loss: {loss.item()} Similarity: {logits_per_image.cpu().detach().numpy().reshape(-1)}", end='\r')


# save and load from pretrained
# model.save_pretrained("vit-bert")
# model = VisionTextDualEncoderModel.from_pretrained("vit-bert")

# inference
# outputs = model(**inputs)
# logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
# probs = logits_per_image.softmax(dim=1) 