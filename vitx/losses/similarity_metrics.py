import torch


# cosine similarity as logits
def cosine_similarity(
    text_embeds: torch.FloatTensor, image_embeds: torch.FloatTensor, logit_scale
):
    logit_scale = logit_scale.exp()
    logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
    logits_per_image = logits_per_text.T

    return logits_per_text, logits_per_image
