import torch
from torch import nn, Tensor

from models.sam2.modeling.sam2_utils import LayerNorm2d


class UncertaintyHead(nn.Module):
    def __init__(self, in_channels: int = 256):
        super().__init__()
        hidden_channels = max(1, in_channels // 4)
        self.proj_in = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.norm = LayerNorm2d(hidden_channels)
        self.act = nn.ReLU(inplace=True)
        self.proj_out = nn.Conv2d(hidden_channels, 1, kernel_size=1)

        # 训练初期让 log_var 约为 0，保证数值更稳定
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj_in(x)
        x = self.norm(x)
        x = self.act(x)
        log_var = self.proj_out(x)
        return log_var
