from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torchvision


class ImageCollateFunction(nn.Module):
    def __init__(self, transform: torchvision.transforms.Compose):
        super(ImageCollateFunction, self).__init__()
        self.transform = transform

    def forward(self, batch: List[tuple]) -> Tuple[torch.FloatTensor, List[str]]:
        batch_size = len(batch)

        # list of transformed images
        transforms = [
            self.transform(batch[i % batch_size][0]).unsqueeze_(0)
            for i in range(batch_size)
        ]

        # list of captions
        captions = [item[1] for item in batch]

        return torch.cat(transforms, 0), captions


class MultiViewCollateFunction(ImageCollateFunction):
    """Generates multiple views for each image in the batch.

    Takes a batch of images as input and transforms each image into two
    different augmentations with the help of random transforms. The images are
    then concatenated such that the output batch is exactly twice the length
    of the input batch.
    """

    def __init__(self, transform: torchvision.transforms.Compose):
        super(MultiViewCollateFunction, self).__init__(transform=transform)

    def forward(self, batch: List[tuple]) -> Tuple[torch.FloatTensor, List[str]]:
        """Turns a batch of tuples into a tuple of batches.

        Args:
            batch (List[tuple]): The input batch.

        Returns:
            Tuple[torch.FloatTensor, List[str]]: A (views, captions) tuple where views is a list of tensors
            with each tensor containing one view of the batch.
        """
        batch_size = len(batch)

        # list of transformed images
        transforms = [
            self.transform(batch[i % batch_size][0]).unsqueeze_(0)
            for i in range(2 * batch_size)
        ]

        # list of captions
        captions = [item[1] for item in batch]

        # stack the transforms
        transforms = torch.stack(
            [
                torch.cat(transforms[:batch_size], 0),
                torch.cat(transforms[batch_size:], 0),
            ],
            dim=1,
        )

        return transforms, captions
