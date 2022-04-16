from typing import Optional

import torch

from ...losses import cosine_similarity

"""  Data Classes for Retrieval Metrics """


@dataclass
class RetrievalMap:
    top_probs_per_text: Optional[torch.FloatTensor] = None
    top_probs_per_image: Optional[torch.FloatTensor] = None
    top_labels_per_text: Optional[torch.FloatTensor] = None
    top_labels_per_image: Optional[torch.FloatTensor] = None


""" Retrieval Metrics Functions """


def retrieval_map(
    logits_per_image: Optional[torch.FloatTensor] = None,
    logits_per_text: Optional[torch.FloatTensor] = None,
    vision_pooled_embeds: Optional[torch.FloatTensor] = None,
    text_pooled_embeds: Optional[torch.FloatTensor] = None,
    topk: Optional[int] = 1,
    logit_scale: Optional[torch.FloatTensor] = torch.ones([]) * 2.6592,
) -> RetrievalMap:
    """Returns Image-Text retrieval map based on image-text similarities or image and text embeddings.

    Args:
        logits_per_image (Optional[torch.FloatTensor], optional): _description_. Defaults to None.
        logits_per_text (Optional[torch.FloatTensor], optional): _description_. Defaults to None.
        vision_pooled_embeds (Optional[torch.FloatTensor], optional): _description_. Defaults to None.
        text_pooled_embeds (Optional[torch.FloatTensor], optional): _description_. Defaults to None.
        topk (Optional[int], optional): _description_. Defaults to 1.
        logit_scale (Optional[torch.FloatTensor], optional): _description_. Defaults to torch.ones([])*2.6592.

    Raises:
        ValueError: _description_

    Returns:
        RetrievalMap: _description_
    """

    if (logits_per_image is None and logits_per_text is None) and (
        text_pooled_embeds is None or vision_pooled_embeds is None
    ):
        raise ValueError(
            "If neither `logits_per_image` nor `logits_per_text` are passed, `text_pooled_embeds` and `vision_pooled_embeds` should be passed to compute the similarities"
        )

    if logits_per_image is not None and text_pooled_embeds is not None:
        vision_pooled_embeds, logits_per_image = cosine_similarity(
            text_embeds=text_pooled_embeds,
            image_embeds=vision_pooled_embeds,
            logit_scale=logit_scale,
        )

    top_probs_per_text, top_labels_per_text = logits_per_text.cpu().topk(topk, dim=-1)
    top_probs_per_image, top_labels_per_image = logits_per_image.cpu().topk(
        topk, dim=-1
    )

    return RetrievalMap(
        top_probs_per_text=top_probs_per_text,
        top_probs_per_image=top_probs_per_image,
        top_labels_per_text=top_labels_per_text,
        top_labels_per_image=top_labels_per_image,
    )
