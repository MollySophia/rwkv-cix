import torch
from typing import Any

class Add(torch.nn.Module):
    # pylint:disable=arguments-differ
    @staticmethod
    def forward(x: Any, y: Any) -> Any:
        if isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor):
            out = torch.add(x, y)
        else:
            out = x + y
        return out

class Subtract(torch.nn.Module):
    # pylint:disable=arguments-differ
    @staticmethod
    def forward(x: Any, y: Any) -> Any:
        if isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor):
            out = torch.sub(x, y)
        else:
            out = x - y
        return out
    
class Neg(torch.nn.Module):
    # pylint:disable=arguments-differ
    @staticmethod
    def forward(x: Any) -> Any:
        out = torch.neg(x)
        return out

class Multiply(torch.nn.Module):
    # pylint:disable=arguments-differ
    @staticmethod
    def forward(x: Any, y: Any) -> Any:
        if isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor):
            out = torch.mul(x, y)
        else:
            out = x * y
        return out

class Tanh(torch.nn.Module):
    # pylint:disable=arguments-differ
    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        out = torch.tanh(x)
        return out

class SiLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.mul = Multiply()

    def forward(self, x: torch.Tensor) -> Any:
        return self.mul(x, self.sigmoid(x))

class Exponential(torch.nn.Module):
    # pylint:disable=arguments-differ
    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x)

class MatMul(torch.nn.Module):
    # pylint:disable=arguments-differ
    @staticmethod
    def forward(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, y)

class Bmm(torch.nn.Module):
    # pylint:disable=arguments-differ
    @staticmethod
    def forward(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.bmm(x, y)

class Split(torch.nn.Module):
    # pylint:disable=arguments-differ
    @staticmethod
    def forward(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return torch.split(x, *args, **kwargs)

class ReLU(torch.nn.Module):
    # pylint:disable=arguments-differ
    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)

class Pow(torch.nn.Module):
    # pylint:disable=arguments-differ
    @staticmethod
    def forward(x: torch.Tensor, y: int) -> torch.Tensor:
        return torch.pow(x, y)

custom_norm_wrapper_src = """
#include <torch/extension.h>
#include <torch/script.h>

torch::Tensor l2norm(torch::Tensor x) {
    return x / (x.norm(2, -1, true) + 1e-7);
}

TORCH_LIBRARY(customop, m) {
    m.def("l2norm", &l2norm);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
}
"""

_ = torch.utils.cpp_extension.load_inline(
    name='extension', cpp_sources=[custom_norm_wrapper_src])

class L2Norm(torch.nn.Module):
    # pylint:disable=arguments-differ
    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        return torch.ops.customop.l2norm(x)

class Lerp(torch.nn.Module):
    def __init__(self, weight: torch.nn.Parameter):
        super().__init__()
        self.mul = Multiply()
        self.add = Add()
        self.weight = weight

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.add(x, self.mul(y, self.weight))

class Gather(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        return torch.gather(x, self.dim, index)

class Sum(torch.nn.Module):
    @staticmethod
    def forward(x: torch.Tensor, dim: int, keepdim: bool) -> torch.Tensor:
        return torch.sum(x, dim, keepdim)