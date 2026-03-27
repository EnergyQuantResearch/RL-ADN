from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


def get_optim_param(optimizer: torch.optim.Optimizer) -> list[torch.Tensor]:
    params_list: list[torch.Tensor] = []
    for params_dict in optimizer.state_dict()["state"].values():
        params_list.extend(value for value in params_dict.values() if isinstance(value, torch.Tensor))
    return params_list


def build_mlp(dims: Iterable[int]) -> nn.Sequential:
    dims = list(dims)
    if len(dims) < 2:
        raise ValueError("build_mlp expects at least an input and output dimension")

    layers: list[nn.Module] = []
    for index in range(len(dims) - 1):
        layers.append(nn.Linear(dims[index], dims[index + 1]))
        if index < len(dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)
