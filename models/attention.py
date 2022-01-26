import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def uniform_attention(q, v):
    total_points = q.shape[1]
    # print(v.shape)
    rep = torch.mean(v,axis=1, keepdim=True)
    # print(rep.shape)
    rep = rep.repeat([1,total_points,1])
    # print(rep.shape)
    # import sys
    # sys.exit()
    return rep

def laplace_attention(q,k,v,scale,normalise):
    #SEE THIS AND UNDERSTAND
    # print("K ", k.shape)
    k = k.unsqueeze(axis = 1)
    # print("af: k ",k.shape)
    # print("q: ",q.shape)
    q = q.unsqueeze(axis = 2)
    # print("af q: ", q.shape)
    unnorm_weights = - torch.abs((k-q)/scale)
    # print(unnorm_weights.shape)
    unnorm_weights = torch.sum(unnorm_weights, axis = -1)
    # print(unnorm_weights.shape)
    # print(unnorm_weights[0])
    if normalise:
        weights = F.softmax(unnorm_weights,dim=2)

    else:
        weight_fn = lambda x: 1 + torch.tanh(x)
        weights = weight_fn(unnorm_weights)
    rep = torch.einsum('bik,bkj->bij', weights, v)
    # print("da?", weights[0])
    # import sys
    # sys.exit()

    return rep

def dot_product_attention(q, k, v, normalise):
    rep_shape = q.shape[1]
    scale = np.sqrt(rep_shape)
    unnorm_weights = torch.einsum('bjk, bik->bij', k , q)/scale
    if normalise:
        weights = F.softmax(unnorm_weights, dim=2)
    else:
        weights = F.sigmoid(unnorm_weights)

    rep = torch.einsum('bik,bkj->bij', weights, v)
    return rep

# def multi_head_attention(q, k, v, num_heads = 8):
#     d_k = q.shape[-1]
#     d_v = v.shape[-1]
#     head_size = d_v / num_heads
#
#     key_initializer =
#     rep =

class Attention(nn.Module):
    def __init__(self, representation, output_sizes, att_type, scale=1,normalise=True,num_heads=8):
        super(Attention,self).__init__()
        self._rep = representation
        self._output_sizes = output_sizes
        self._type = att_type
        self._scale = scale
        self._normalise = normalise
        if self._type == "multihead":
            self._num_heads = num_heads

        if self._rep == 'mlp':
            self.linear_layers_list = nn.ModuleList()
            for i in range(len(output_sizes) - 1):
                layer = nn.Linear(output_sizes[i], output_sizes[i + 1])
                nn.init.xavier_uniform_(layer.weight)
                self.linear_layers_list.append(layer)

            self.linear_layers_list2 = nn.ModuleList()
            for i in range(len(output_sizes) - 1):
                layer = nn.Linear(output_sizes[i], output_sizes[i + 1])
                nn.init.xavier_uniform_(layer.weight)
                self.linear_layers_list2.append(layer)
            self._output_sizes = output_sizes

        if self._type == "multihead":
            # print(self._output_sizes[-1], self._num_heads, "what went wrong?")
            self.multihead_attn = nn.MultiheadAttention(self._output_sizes[-1],self._num_heads)

    def forward(self,x1,x2, r, mask = None):
        # print(x1.shape)
        # print(x2.shape)
        # print(r.shape)

        if self._rep == 'identity':
            k,q = (x1,x2)
        elif self._rep  == 'mlp':
            batch_size, set_size, filter_size = x1.shape  # encoder input:  torch.Size([64, 784, 3])
            # print("bef mlp: ", x1.shape, x2.shape)
            k = x1.reshape(batch_size * set_size, -1)
            # print("x shape: ", x.shape) #x shape:  torch.Size([50176, 3])
            for i, linear in enumerate(self.linear_layers_list[:-1]):
                k = torch.relu(linear(k))
            k = self.linear_layers_list[-1](k)
            # print("k: ", k.shape)
            k = k.view(batch_size, set_size, -1)
            # print("k: ", k.shape)

            batch_size, set_size, filter_size = x2.shape
            q = x2.reshape(batch_size * set_size, -1)
            # print('q shape: ', q.shape)
            # print(self.linear_layers_list2)
            # print("x shape: ", x.shape) #x shape:  torch.Size([50176, 3])
            for i, linear in enumerate(self.linear_layers_list2[:-1]):
                q = torch.relu(linear(q))
            # print("q : ", q.shape)
            q = self.linear_layers_list2[-1](q)
            # print("q : ", q.shape)
            q = q.view(batch_size, set_size, -1)
            # print("after mlp: ", k.shape, q.shape, r.shape, mask.shape)


        if self._type == 'uniform':
            rep = uniform_attention(q,r)
        elif self._type == 'laplace':
            rep = laplace_attention(q,k,r, self._scale, self._normalise)
        elif self._type == 'dot_product':
            rep = dot_product_attention(q,k,r,self._normalise)
        elif self._type == 'multihead':
            q =torch.transpose(q,0,1)
            k =torch.transpose(k,0,1)
            r = torch.transpose(r,0,1)
            # print(q.shape)
            # print(k.shape)
            # print(r.shape)
            rep, rep_wts = self.multihead_attn(q,k,r)
            # print(rep.shape)
            rep = torch.transpose(rep, 0,1)

        # print("attention rep shape: ", rep.shape)

        return rep
