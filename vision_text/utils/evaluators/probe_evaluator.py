from typing import Optional

import torch
import pytorch_lightning as pl


class ProbeEvaluator(pl.Trainer):

    def __init__(
        self,
        *args,
        **kwargs
    ):

        super(ProbeEvaluator, self).__init__(*args, **kwargs)