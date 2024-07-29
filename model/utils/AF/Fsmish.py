"""
Script based on:
Wang, Xueliang, Honge Ren, and Achuan Wang.
 "Smish: A Novel Activation Function for Deep Learning Methods.
 " Electronics 11.4 (2022): 540.
"""

# import pytorch
import torch
import torch.nn.functional as F


@torch.jit.script
def smish(input):
    """
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(sigmoid(x))))
    See additional documentation for mish class.
    """
    input = 1+torch.sigmoid(input)
    # print('input1',input.size())

    input = torch.log(input)
    # print('input2',input.size())

    input = input * torch.tanh(input)
    # print('input3',input.size())

    return input

    # return input * torch.tanh(torch.log(1+torch.sigmoid(input)))