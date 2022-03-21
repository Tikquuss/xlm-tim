from tkinter.messagebox import NO
from .quantizer_function import QuantizerFunction
import torch.nn as nn

class QIdentity(nn.Module):
    def __init__(self, input_dims,  codebook_size, method, n_factors=[1,2,4]):
        super(QIdentity, self).__init__()
        self.q = nn.Identity()

    def forward(self, state):
        return self.q(state), 0, None

def getQuantizerFunction(input_dims,  codebook_size, q_method = None, n_factors=[1,2,4]):
    if q_method is None or q_method == "Original":
        return QIdentity(input_dims,  codebook_size, q_method, n_factors)
    else :
        return QuantizerFunction(input_dims,  codebook_size, q_method, n_factors)