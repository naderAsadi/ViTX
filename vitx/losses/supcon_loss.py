from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def supcon_loss(
    features: torch.FloatTensor,
    temperature: float,
    labels: Optional[torch.FloatTensor] = None,
    base_temperature: Optional[float] = 0.07,
    contrast_mode: Optional[str] = "all",
):
    if len(features.shape) < 3:
        raise ValueError(
            "`features` needs to be [bsz, n_views, ...],"
            "at least 3 dimensions are required"
        )
    if len(features.shape) > 3:
        features = features.view(features.shape[0], features.shape[1], -1)

    batch_size = features.shape[0]
    if labels is None:
        mask = torch.eye(batch_size, dtype=torch.float32)
    else:
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError("Num of labels does not match num of features")
        mask = torch.eq(labels, labels.T).float()

    contrast_count = features.shape[1]
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
    if contrast_mode == "one":
        anchor_feature = features[:, 0]
        anchor_count = 1
    elif contrast_mode == "all":
        anchor_feature = contrast_feature
        anchor_count = contrast_count
    else:
        raise ValueError("Unknown contrast mode: {}".format(contrast_mode))

    # compute logits
    anchor_dot_contrast = torch.div(
        torch.matmul(anchor_feature, contrast_feature.T), temperature
    )
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # tile mask
    mask = mask.repeat(anchor_count, contrast_count)
    # mask-out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size * anchor_count).view(-1, 1),
        0,
    )
    mask = mask * logits_mask

    mask = mask.type_as(features)
    logits_mask = logits_mask.type_as(features)

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    # loss
    loss = -(temperature / base_temperature) * mean_log_prob_pos
    loss = loss.view(anchor_count, batch_size).mean()

    return loss
