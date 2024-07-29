from torch import nn

import model.utils.AF.Fsmish as Func


class Smish(nn.Module):
    def __init__(self):
        """
        Init method.
        """
        super().__init__()

    def forward(self, input):
        """
        Forward pass of the function.
        """
        return Func.smish(input)