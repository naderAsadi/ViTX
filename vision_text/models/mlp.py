from typing import Optional
import torch
import torch.nn as nn

from . import register_model


@register_model("linear")
class LinearClassifier(nn.Module):
    """Linear classifier"""

    def __init__(self, input_dim, output_dim, **kwargs):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


def add_linear(input_dim: int, output_dim: int, batch_norm: bool, relu: bool):
    layers = []
    layers.append(nn.Linear(input_dim, output_dim))
    if batch_norm:
        layers.append(nn.BatchNorm1d(output_dim))
    if relu:
        layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


@register_model("mlp")
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: Optional[int] = 2,
        batch_norm: Optional[bool] = False,
        **kwargs
    ):
        super(MLP, self).__init__()
        self.layers = self._make_layers(
            input_dim, hidden_dim, output_dim, n_layers, batch_norm
        )

    def _make_layers(self, input_dim, hidden_dim, output_dim, n_layers, batch_norm):
        dims = [input_dim] + (n_layers - 1) * [hidden_dim] + [output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(
                add_linear(
                    input_dim=dims[i],
                    output_dim=dims[i + 1],
                    batch_norm=batch_norm,
                    relu=(i < len(dims) - 2),
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
