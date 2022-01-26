import torch
import torch.nn as nn

from models.building_blocks import get_the_network_linear_list
from models.building_blocks import forward_pass_linear_layer_relu
import torch.nn.functional as F


class ContextToLatentDistribution(nn.Module):
    '''
    Transform the encoded representation to mean and log_variance
    '''

    def __init__(self, representation_size):
        super(ContextToLatentDistribution, self).__init__()
        self.mean_layer = nn.Linear(representation_size, representation_size)
        self.log_variance_layer = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        return self.mean_layer(x), self.log_variance_layer(x)


class ContextToEvidentialLatentDistribution(nn.Module):
    '''
    Transform the encoded representation to mean and log_variance
    '''

    def __init__(self, representation_size):
        super(ContextToEvidentialLatentDistribution, self).__init__()
        # self.gamma = nn.Sequential(nn.ReLU(), nn.Linear(representation_size, representation_size), nn.ReLU(),
        #                            nn.Linear(representation_size, representation_size))
        # self.v = nn.Sequential(nn.ReLU(), nn.Linear(representation_size, representation_size), nn.ReLU(),
        #                        nn.Linear(representation_size, representation_size))
        # self.alpha = nn.Sequential(nn.ReLU(), nn.Linear(representation_size, representation_size), nn.ReLU(),
        #                            nn.Linear(representation_size, representation_size))
        # self.beta = nn.Sequential(nn.ReLU(), nn.Linear(representation_size, representation_size), nn.ReLU(),
        #                           nn.Linear(representation_size, representation_size))
        self.gamma = nn.Sequential(nn.ReLU(),
                                   nn.Linear(representation_size, representation_size))
        self.v = nn.Sequential(nn.ReLU(),
                               nn.Linear(representation_size, representation_size))
        self.alpha = nn.Sequential(nn.ReLU(),
                                   nn.Linear(representation_size, representation_size))
        self.beta = nn.Sequential(nn.ReLU(),
                                  nn.Linear(representation_size, representation_size))

    def forward(self, x):
        return self.gamma(x), self.v(x), self.alpha(x), self.beta(x),


class ANPDeterministicEncoder(nn.Module):
    '''
    The encoder
    '''

    def __init__(self, output_sizes, attention, args=None):
        '''
        ANP Encoder
        Can realize the CNP with identity attention
        :param output_sizes: Controls the output size of the encoder representation (R)
        '''
        super(ANPDeterministicEncoder, self).__init__()

        self.linear_layers_list = get_the_network_linear_list(output_sizes)

        self._attention = attention
        self._args = args

    def forward(self, context_x, context_y, target_x, mask=None):
        encoder_input = torch.cat([context_x, context_y], axis=-1)

        # Get the shape of input
        batch_size, set_size, filter_size = encoder_input.shape
        x = forward_pass_linear_layer_relu(encoder_input, self.linear_layers_list)

        x = x.view(batch_size, set_size, -1)

        representation = self._attention(context_x, target_x, x)

        return representation


class CNPDeterministicEncoder_contrastive(nn.Module):
    '''
    The encoder
    '''

    def __init__(self, output_sizes, args=None):
        '''
        CNP Encoder
        :param output_sizes: Controls the output size of the encoder representation (R)
        '''
        super(CNPDeterministicEncoder_contrastive, self).__init__()

        self.linear_layers_list = get_the_network_linear_list(output_sizes)

        self._args = args

    def forward(self, context_x, context_y, target_x, mask=None):
        encoder_input = torch.cat([context_x, context_y], axis=-1)

        # Get the shape of input
        batch_size, set_size, filter_size = encoder_input.shape
        x = forward_pass_linear_layer_relu(encoder_input, self.linear_layers_list)

        x = x.view(batch_size, set_size, -1)

        return x


class ANPDecoder(nn.Module):
    '''
    The Decoder
    '''

    def __init__(self, output_sizes, args=None):
        super(ANPDecoder, self).__init__()
        self.linear_layers_list = get_the_network_linear_list(output_sizes)
        self._channels = args.channels

    def forward(self, representation, target_x):
        batch_size, set_size, d = target_x.shape
        # representation = representation.unsqueeze(1).repeat([1,set_size,1])
        # print("rep: ", representation.shape)
        input_data = torch.cat((representation, target_x), dim=-1)

        x = forward_pass_linear_layer_relu(input_data, self.linear_layers_list)

        out = x.view(batch_size, set_size, -1)

        mu, log_sigma = torch.split(out, self._channels, dim=-1)
        sigma = 0.1 + 0.9 * torch.nn.functional.softplus(log_sigma)
        # sigma = torch.exp(log_sigma)
        dist = torch.distributions.normal.Normal(loc=mu, scale=sigma)
        return dist, mu, sigma


class ANPLatentEncoder(nn.Module):
    '''
    The encoder
    '''

    def __init__(self, output_sizes):
        '''
        CNP Encoder
        :param output_sizes: Controls the output size of the encoder representation (R)
        '''
        super(ANPLatentEncoder, self).__init__()
        self.linear_layers_list = get_the_network_linear_list(output_sizes)
        self._latent = ContextToLatentDistribution(output_sizes[-1])

    def forward(self, context_x, context_y):
        encoder_input = torch.cat([context_x, context_y], axis=-1)

        # Get the shape of input and reshape to parllelise across observations
        # batch size, number of context points, context_point_shape (i.e may be just 3)
        batch_size, set_size, filter_size = encoder_input.shape
        x = forward_pass_linear_layer_relu(encoder_input, self.linear_layers_list)
        x = x.view(batch_size, set_size, -1)

        representation = x.mean(dim=1)

        representation_latent = self._latent(representation)
        mean, log_std = representation_latent
        std = 0.1 + 0.9 * torch.sigmoid(log_std)
        # std = 0.1 + 0.9 * torch.nn.functional.softplus(log_std)

        dist = torch.distributions.normal.Normal(loc=mean, scale=std)

        return dist, mean, std


class ANPEvidentialLatentEncoder(nn.Module):
    '''
    The encoder
    '''

    def __init__(self, output_sizes, args=None):
        '''
        CNP Encoder
        :param output_sizes: Controls the output size of the encoder representation (R)
        '''
        super(ANPEvidentialLatentEncoder, self).__init__()
        self.linear_layers_list = get_the_network_linear_list(output_sizes)
        self._evidential_latent = ContextToEvidentialLatentDistribution(output_sizes[-1])

        if args == None:
            print("Pass the arguments to the evidential latent decoder")
            raise NotImplementedError

        self._channels = args.channels
        self._ev_lat_beta_min = args.ev_lat_beta_min
        self._ev_lat_alpha_max = args.ev_lat_alpha_max
        self._ev_lat_v_max = args.ev_lat_v_max

    def evidence(self, x):
        # return tf.exp(x)
        return F.softplus(x)

    def get_representation(self, context_x, context_y):
        # Get the shape of input and reshape to parllelise across observations
        # batch size, number of context points, context_point_shape (i.e may be just 3)
        encoder_input = torch.cat([context_x, context_y], axis=-1)

        batch_size, set_size, filter_size = encoder_input.shape
        x = forward_pass_linear_layer_relu(encoder_input, self.linear_layers_list)

        x = x.view(batch_size, set_size, -1)

        representation = x.mean(dim=1)
        return representation

    def forward(self, context_x, context_y):
        encoder_input = torch.cat([context_x, context_y], axis=-1)

        # Get the shape of input and reshape to parllelise across observations
        # batch size, number of context points, context_point_shape (i.e may be just 3)
        batch_size, set_size, filter_size = encoder_input.shape
        x = forward_pass_linear_layer_relu(encoder_input, self.linear_layers_list)

        x = x.view(batch_size, set_size, -1)

        representation = x.mean(dim=1)

        pred, logv, logalpha, logbeta, = self._evidential_latent(representation)  # torch.split(out, self._channels, dim=-1)
        # v = torch.exp(logv) #self.evidence(logv) # + 1
        v = self.evidence(logv) # + 1
        #alpha = torch.exp(logalpha) #self.evidence(logalpha)
        alpha = self.evidence(logalpha)
        alpha = alpha + 1
        # beta = logbeta
        beta = self.evidence(logbeta) 
        # beta = torch.sigmoid(logbeta) # NP implementation code

        # The constraints
        alpha_thr = self._ev_lat_alpha_max * torch.ones(alpha.shape).to(alpha.device)
        alpha = torch.min(alpha, alpha_thr)
        v_thr = self._ev_lat_v_max * torch.ones(v.shape).to(v.device)
        v = torch.min(v, v_thr)
        beta_min = self._ev_lat_beta_min * torch.ones(beta.shape).to(beta.device)
        beta = beta + beta_min #+ 0.9 * torch.sigmoid(beta) #beta + 0.2
        



        gamma_dist = torch.distributions.gamma.Gamma(alpha, beta)
        # gamma_sample = gamma_dist.rsample()
        sample_inv_gamma = 1 / gamma_dist.rsample()  
        std_mu = sample_inv_gamma / v
        normal_dist = torch.distributions.normal.Normal(pred, std_mu )
        sample_normal = normal_dist.rsample()

        std_dist = sample_inv_gamma
        std_dist = torch.sqrt(std_dist)
        dist = torch.distributions.normal.Normal(sample_normal,  std_dist )
        
        #NP realization from ENP-L if needed, simplified
        #dist = torch.distributions.normal.Normal(pred,  beta )

        return dist, (pred, v, alpha, beta), (sample_normal, sample_inv_gamma)


class ANPEvidentialDecoder(nn.Module):
    '''
    The Decoder
    '''

    def __init__(self, output_sizes, args=None):
        super(ANPEvidentialDecoder, self).__init__()
        # print("output sizes: ",  output_sizes)
        self.linear_layers_list = get_the_network_linear_list(output_sizes[:-1])
        if args == None:
            print("pass args to ANPEvidentialDecoder in np_blocs.py")
            raise NotImplementedError
        self._channels = args.channels
        self._ev_dec_beta_min = args.ev_dec_beta_min
        self._ev_dec_alpha_max = args.ev_dec_alpha_max
        self._ev_dec_v_max = args.ev_dec_v_max

        self.transform_gamma = nn.Sequential(nn.ReLU(), nn.Linear(output_sizes[-2], 64), nn.ReLU(),
                                             nn.Linear(64, args.channels))
        self.transform_v = nn.Sequential(nn.ReLU(), nn.Linear(output_sizes[-2], 64), nn.ReLU(),
                                             nn.Linear(64, args.channels))
        self.transform_alpha = nn.Sequential(nn.ReLU(), nn.Linear(output_sizes[-2], 64), nn.ReLU(),
                                             nn.Linear(64, args.channels))
        self.transform_beta = nn.Sequential(nn.ReLU(), nn.Linear(output_sizes[-2], 64), nn.ReLU(),
                                             nn.Linear(64, args.channels))
    def evidence(self, x):
        return F.softplus(x)

    def forward(self, representation, target_x):
        batch_size, set_size, d = target_x.shape
        input_data = torch.cat((representation, target_x), dim=-1)

        x = forward_pass_linear_layer_relu(input_data, self.linear_layers_list)


        gamma = self.transform_gamma(x).view(batch_size,set_size,-1)
        logv = self.transform_v(x).view(batch_size,set_size,-1)
        logalpha = self.transform_alpha(x).view(batch_size,set_size,-1)
        logbeta = self.transform_beta(x).view(batch_size,set_size,-1)


        v = self.evidence(logv)  # + 1.0
        alpha = self.evidence(logalpha)
        alpha = alpha + 1
        beta = self.evidence(logbeta)

        # The constraints
        alpha_thr = self._ev_dec_alpha_max * torch.ones(alpha.shape).to(alpha.device)
        alpha = torch.min(alpha, alpha_thr)
        v_thr = self._ev_dec_v_max * torch.ones(v.shape).to(v.device)
        v = torch.min(v, v_thr)
        beta_min = self._ev_dec_beta_min * torch.ones(beta.shape).to(beta.device)
        beta = beta + beta_min
        return gamma, v, alpha, beta


