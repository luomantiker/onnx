import torch
import torch.nn as nn

from cuda_op import custom_add


class CustomAddFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return custom_add(x, y)

    @staticmethod
    def symbolic(g, x, y):
        return g.op("custom_domain::CustomAdd", x, y)


class ModelWithCuda(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return CustomAddFunction.apply(x, y)
