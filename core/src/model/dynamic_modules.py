import torch
import torch.nn as nn

class HierachicalModuleSelection(nn.Module):
    def __init__(self, query_dim, n_specialists, key_dim, num_heads=4):
        super().__init__()
        assert query_dim % num_heads == 0, 'dim must be a multiple of n_heads'
        key_dim = query_dim if key_dim is None else key_dim
        #self.specialists_keys = nn.Linear(key_dim, n_specialists, bias=False)
        self.specialists_keys = torch.nn.Parameter(torch.randn(n_specialists, key_dim), requires_grad=True) # (n_specialists, key_dim)
        self.specialists_attention = torch.nn.MultiheadAttention(embed_dim=query_dim, num_heads=num_heads, kdim=key_dim, vdim=key_dim, batch_first=True)
        
        self.softmax_temperature = torch.nn.Parameter(torch.randn(1).squeeze(), requires_grad=True) # (,)

    def forward(self, tensor, states, hard_gumbel_softmax=True):
        """
        tensor : (batch_size, seq_len, dim)
        states : n_specialists x (batch_size, seq_len, dim)
        """
        # The same key is use for all sequence of the batch
        #kv = self.specialists_keys.weight.repeat(tensor.size(0), 1, 1) # (batch_size, n_specialists, keys_dim)
        kv = self.specialists_keys.repeat(tensor.size(0), 1, 1) # (batch_size, n_specialists, keys_dim)
        _, att_scores = self.specialists_attention(query=tensor, key=kv, value=kv) # (batch_size, seq_len, n_specialists)
        att_scores = nn.functional.gumbel_softmax(att_scores, hard=hard_gumbel_softmax, dim=-1, tau=self.softmax_temperature) # (batch_size, seq_len, n_specialists)
        states = torch.stack(states, 1) # (batch_size, n_specialists, seq_len, dim)
        states = states.permute(0, 2, 1, 3) # (batch_size, seq_len, n_specialists, dim)
        tensor = states * att_scores.unsqueeze(-1) # (batch_size, seq_len, n_specialists, dim)
        #tensor = tensor[tensor != 0.] # (batch_size, seq_len, dim)
        tensor = tensor.sum(2) # (batch_size, seq_len, dim)
        return tensor

class HMSIdentity(nn.Module):
    def __init__(self, query_dim, n_specialists, key_dim, num_heads):
        super().__init__()
        self.q = nn.Identity()

    def forward(self, tensor, states):
        return self.q(tensor)