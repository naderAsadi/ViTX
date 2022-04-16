from dataclasses import dataclass
from typing import Optional

import torch

from ...losses import cosine_similarity

"""  Data Classes for Retrieval Metrics """


@dataclass
class RetrievalMap:
    acc_per_text: Optional[torch.FloatTensor] = None
    acc_per_image: Optional[torch.FloatTensor] = None
    top_probs_per_text: Optional[torch.FloatTensor] = None
    top_probs_per_image: Optional[torch.FloatTensor] = None
    top_labels_per_text: Optional[torch.FloatTensor] = None
    top_labels_per_image: Optional[torch.FloatTensor] = None


""" Retrieval Metrics Functions """


def get_retrieval_map(
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

    if vision_pooled_embeds is not None and text_pooled_embeds is not None:
        logits_per_text, logits_per_image = cosine_similarity(
            text_embeds=text_pooled_embeds,
            image_embeds=vision_pooled_embeds,
            logit_scale=logit_scale,
        )

    retrieval_map = RetrievalMap()

    if logits_per_text is not None:
        # get softmax per text
        logits_per_text = logits_per_text.softmax(dim=-1)
        # get topk predictions and their indices
        (
            retrieval_map.top_probs_per_text,
            retrieval_map.top_labels_per_text,
        ) = logits_per_text.cpu().topk(topk, dim=-1)
        # create target labels
        labels_per_text = torch.arange(0, logits_per_text.size(0))
        # calculate image retrieval acc per text
        retrieval_map.acc_per_text = sum(
            label in retrieval_map.top_labels_per_text
            for idx, label in enumerate(labels_per_text)
        ) / labels_per_text.size(0)

    if logits_per_image is not None:
        # get softmax per text
        logits_per_image = logits_per_image.softmax(dim=-1)
        # get topk predictions and their indices
        (
            retrieval_map.top_probs_per_image,
            retrieval_map.top_labels_per_image,
        ) = logits_per_image.cpu().topk(topk, dim=-1)
        # create target labels
        labels_per_image = torch.arange(0, logits_per_image.size(0))
        # calculate text retrieval acc per image
        retrieval_map.acc_per_image = sum(
            label in retrieval_map.top_labels_per_image
            for idx, label in enumerate(labels_per_image)
        ) / labels_per_image.size(0)

    return retrieval_map
