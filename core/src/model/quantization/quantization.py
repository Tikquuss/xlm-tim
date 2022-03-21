import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
#import distributed as dist_fn

import torch
from torch import nn, einsum
import torch.nn.functional as F

from scipy.cluster.vq import kmeans2
import random
#from torchviz import make_dot

from logging import getLogger
logger = getLogger()


class Quantize(nn.Module):
	"""
	Neural Discrete Representation Learning, van den Oord et al. 2017
	https://arxiv.org/abs/1711.00937
	Follows the original DeepMind implementation
	https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
	https://github.com/deepmind/sonnet/blob/v2/examples/vqvae_example.ipynb
	"""
	def __init__(self, dim, n_embed, groups,Using_projection_layer=False):
		super().__init__()

		num_hiddens=dim
		embedding_dim=dim
		self.embedding_dim = embedding_dim
		self.n_embed = n_embed
		self.groups = groups

		self.kld_scale = 10.0


		self.Using_projection_layer=Using_projection_layer #if a projection layer should be used after quantization

		self.proj = nn.Linear(embedding_dim, embedding_dim)
		
		self.embed = nn.Embedding(n_embed, embedding_dim//groups)

		self.register_buffer('data_initialized', torch.zeros(1))

	def forward(self, z):
		#####input is batch size (B)X Number of units Xembedding size

		B, N,D= z.size()
		#W = 1

		# project and flatten out space, so (B, C, H, W) -> (B*H*W, C)
		#z_e = self.proj(z)
		#z_e = z_e.permute(0, 2, 3, 1) # make (B, H, W, C)
		#z_e = z.reshape((B, H, self.groups, self.embedding_dim//self.groups)).reshape((B, H*self.groups, self.embedding_dim//self.groups))
		
		z_e = z.reshape((B, N, self.groups, self.embedding_dim//self.groups)).reshape((B*N*self.groups, self.embedding_dim//self.groups))
		
		flatten = z_e.reshape(-1, self.embedding_dim//self.groups)

		#flatten = flatten.reshape((flatten.shape[0], self.groups, self.embedding_dim//self.groups)).reshape((flatten.shape[0] * self.groups, self.embedding_dim//self.groups))

		# DeepMind def does not do this but I find I have to... ;\
		if self.training and self.data_initialized.item() == 0:
			logger.info('running kmeans!!') # data driven initialization for the embeddings
			rp = torch.randint(0,flatten.size(0),(20000,))###batch size is small in RIM, up sampling here for clustering 
			kd = kmeans2(flatten[rp].data.cpu().numpy(), self.n_embed, minit='points')
			self.embed.weight.data.copy_(torch.from_numpy(kd[0]))
			self.data_initialized.fill_(1)
			# TODO: this won't work in multi-GPU setups

		
		dist = (
			flatten.pow(2).sum(1, keepdim=True)
			- 2 * flatten @ self.embed.weight.t()
			+ self.embed.weight.pow(2).sum(1, keepdim=True).t()
		)

		
		_, ind = (-dist).max(1)
		
		#ind = ind.view(B,self.groups)
		
		# vector quantization cost that trains the embedding vectors
		z_q = self.embed_code(ind) # (B, H, W, C)
		commitment_cost = 0.25
		diff = commitment_cost * (z_q.detach() - z_e).pow(2).mean() + (z_q - z_e.detach()).pow(2).mean()
		diff *= self.kld_scale

		z_q = z_e + (z_q - z_e).detach() # noop in forward pass, straight-through gradient estimator in backward pass

		z_q = z_q.reshape((B,N,self.groups, self.embedding_dim//self.groups)).reshape((B,N,self.embedding_dim))

		if self.Using_projection_layer:
			###linear projection of quantized embeddings after quantization
			z_q_projected=[]
			for i in range(N):
				z_q_projected.append(self.proj(z_q[:,i,:]))

			z_q=torch.stack(z_q_projected,1)


		ind = ind.reshape(B,N,self.groups).view(N,B*self.groups)
		
		if random.uniform(0,1) < 0.000001:
			logger.info('encoded ind',ind.view(N,B,self.groups)[:,0,:])
			logger.info('before', z[0])
			logger.info('after', z_q[0])
			logger.info('extra loss on layer', diff)
		
		return z_q, diff, ind.view(N,B,self.groups)


	def embed_code(self, embed_id):
		return F.embedding(embed_id, self.embed.weight)