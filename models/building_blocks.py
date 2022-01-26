import torch.nn as nn
import torch


def get_the_network_linear_list(output_sizes):
    '''
    Make FC linear layers with shapes defined by output_sizes list.
    '''
    linear_layers_list = nn.ModuleList()
    for i in range(len(output_sizes) - 1):
        linear_layers_list.append(nn.Linear(output_sizes[i], output_sizes[i + 1]))
        if i <len(output_sizes) - 2: linear_layers_list.append(nn.ReLU())
    return linear_layers_list

def forward_pass_linear_layer_relu(x, linear_layers_list):
    '''
    x: input to the function: Is of shape batch_size , set_size, dimension of input
    The input is first changed to shape (batch_size*set_size, -1)
    Then it is passed through the linear_layers of the network with relu activation function
    '''
    # batch size, number of context points, context_point_shape (i.e may be just 3)
    batch_size, set_size, filter_size = x.shape
    x = x.view(batch_size * set_size, -1)
    for i, linear in enumerate(linear_layers_list[:-1]):
        x = torch.relu(linear(x))
    # print("linear layer list: ", linear_layers_list)
    x = linear_layers_list[-1](x)
    return x

