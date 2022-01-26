import numpy as np

from models.np_blocks import ANPDeterministicEncoder
from models.np_blocks import ANPLatentEncoder
from models.np_blocks import ANPDecoder
from models.np_template import np_model
import torch


class ANPModel(np_model):

    def __init__(self,
                 latent_encoder_output_size,
                 deterministic_encoder_output_size,
                 decoder_output_size,
                 args = None,
                 attention=None, ):
        super(ANPModel, self).__init__(args)
        if args == None:
            raise NotImplementedError
        np_model.__init__(self, args)

        if self._use_deterministic_path:  # CNP or ANP Modelx
            self._deterministic_encoder = ANPDeterministicEncoder(deterministic_encoder_output_size, attention)

        if self._use_latent_path:
            self._latent_encoder = ANPLatentEncoder(latent_encoder_output_size)

        self._decoder = ANPDecoder(decoder_output_size, args=args)
        # print("Decoder: ", self._decoder)

        # print("The NP Model")


    def forward(self, query, target_y=None):
        '''
        :param query: ( (context_x, context_y), target_x ).
        :param target_y:
        :return:
        '''
        # print("Forward ANP")
        (context_x, context_y), target_x = query

        if self._use_latent_path:
            # print("Forward ANP Latent")
            ##PRIOR
            ctx_lat_dist, ctx_lat_mean, ctx_lat_var = self._latent_encoder(context_x,context_y)

            # During training, we have target_y. We use the target for latent encoder
            if target_y is None:
                # sample = torch.randn(ctx_lat_mean.shape).to(ctx_lat_mean.device)
                # latent_rep_sample = ctx_lat_mean + (ctx_lat_var * sample)

                latent_rep_sample = ctx_lat_dist.rsample()
            else:
                ##POSTERIOR
                tar_lat_dist, tar_lat_mean, tar_lat_var = self._latent_encoder(target_x, target_y)

                # sample = torch.randn(tar_lat_mean.shape).to(tar_lat_mean.device)
                # latent_rep_sample = tar_lat_mean + (tar_lat_var * sample)
                latent_rep_sample = tar_lat_dist.rsample()

            batch_size, set_size, _ = target_x.shape
            latent_rep_sample = latent_rep_sample.unsqueeze(1).repeat([1, set_size, 1])

        if self._use_deterministic_path:
            deterministic_rep = self._deterministic_encoder(context_x, context_y, target_x)

        if self._use_deterministic_path and self._use_latent_path:
            representation = torch.cat((deterministic_rep, latent_rep_sample), dim=-1)
        elif self._use_latent_path:
            representation = latent_rep_sample
        elif self._use_deterministic_path:
            representation = deterministic_rep
        else:
            raise ValueError("You need at least one path for the encoder")

        dist, mu, sigma = self._decoder(representation, target_x)


        # Training tasks
        if target_y is not None:
            # print("Training tasks")

            log_likelihood = dist.log_prob(target_y)

            # print("log likelihood: ", log_likelihood.shape, "targety: ", target_y.shape)
            recons_loss = -torch.mean(log_likelihood)
            kl_loss = None
            if self._use_latent_path:
                dist_1 = torch.distributions.Normal(ctx_lat_mean, ctx_lat_var)
                dist_2 = torch.distributions.Normal(tar_lat_mean, tar_lat_var)
                kl_loss_dir = torch.distributions.kl_divergence(dist_2, dist_1)
                # kl_los =  torch.log(ctx_lat_var) -  torch.log(tar_lat_var) \
                #           + 0.5 * ( tar_lat_var**2 / ctx_lat_var**2 + \
                #                 (tar_lat_mean - ctx_lat_mean) ** 2 / ctx_lat_var**2 - 1)
                #
                # # Consider shape and think here
                # kl_loss = torch.mean(kl_los)
                # print("kl: ", torch.mean(kl_loss_dir))
                # print("cur kl: ", kl_loss)
                loss = recons_loss + torch.mean(kl_loss_dir)
            else:
                loss = recons_loss

        else:
            recons_loss = None
            kl_loss = None
            loss = None

        return dist, mu, sigma, recons_loss, kl_loss, loss

    def test_get_encoder_representation(self, query):
        '''
        :param query: ( (context_x, context_y), target_x ).
        :param target_y:
        :param epoch:
        :return:
        '''
        (context_x, context_y), target_x = query

        if self._use_latent_path:
            ##PRIOR
            ctx_lat_dist, ctx_lat_mean, ctx_lat_var = self._latent_encoder(context_x,context_y)

            sample = torch.randn(ctx_lat_mean.shape).to(ctx_lat_mean.device)
            latent_rep_sample = ctx_lat_mean + (ctx_lat_var * sample)

            batch_size, set_size, _ = target_x.shape
            latent_rep_sample = latent_rep_sample.unsqueeze(1).repeat([1, set_size, 1])

        if self._use_deterministic_path:
            deterministic_rep = self._deterministic_encoder(context_x, context_y, target_x)

        if self._use_deterministic_path and self._use_latent_path:
            representation = torch.cat((deterministic_rep, latent_rep_sample), dim=-1)
        elif self._use_latent_path:
            representation = latent_rep_sample
        elif self._use_deterministic_path:
            representation = deterministic_rep
        else:
            raise ValueError("You need at least one path for the encoder")

        representation = torch.mean(representation,dim=1)
        return representation



from models.np_blocks import ANPEvidentialDecoder
from trainingHelpers.lossFunctions import calculate_evidential_loss_constraints
from trainingHelpers.lossFunctions import calc_ev_krnl_reg

from models.np_blocks import ANPEvidentialLatentEncoder

class Evd_det_model(np_model):

    def __init__(self,
                 latent_encoder_output_size,
                 deterministic_encoder_output_size,
                 decoder_output_size,
                 args=None,
                 attention=None, ):
        super(Evd_det_model, self).__init__(args)
        np_model.__init__(self, args)

        if args == None:
            raise NotImplementedError
        self.args = args

        if self._use_deterministic_path:  # CNP or ANP Model
            self._deterministic_encoder = ANPDeterministicEncoder(deterministic_encoder_output_size, attention)

        if self._use_latent_path:
            raise NotImplementedError

        self._evidential_decoder = ANPEvidentialDecoder(decoder_output_size, args=args)

    def forward(self, query, target_y=None, epoch=0, it=0):
        '''
        :param query: ( (context_x, context_y), target_x ).
        :param target_y:
        :param it:
        :return:
        '''
        (context_x, context_y), target_x = query

        # Skipped use latent path
        if self._use_latent_path:
            raise NotImplementedError

        if self._use_deterministic_path:
            deterministic_rep = self._deterministic_encoder(context_x, context_y, target_x)
            representation = deterministic_rep
        else:
            raise ValueError("You need The deterministic path for the encoder")

        mu, v, alpha, beta = self._evidential_decoder(representation, target_x)

        recons_loss = None
        kl_loss = None
        loss = None
        # Training tasks
        if target_y is not None:
            loss = torch.zeros(size=(1,), device=target_y.device)
            if self._use_deterministic_path or self._use_latent_path:
                loss_det, debug_save_logging_dict = calculate_evidential_loss_constraints(it, target_y, mu, v, alpha,
                                                                                          beta,
                                                                                          lambda_coef=self.args.nig_nll_reg_coef)

                if self.args.nig_nll_ker_reg_coef > 0:
                    dist_based_reg = calc_ev_krnl_reg(target_x, context_x, v, lambda_ker=self.args.nig_nll_ker_reg_coef)
                    loss += dist_based_reg
                loss += loss_det

        df = 2 * alpha
        loc = mu
        scale = torch.sqrt(beta * (1 + v)/ v / alpha)
        dist = torch.distributions.studentT.StudentT(df=df, loc=loc, scale=scale)
        return dist, recons_loss, kl_loss, loss, mu, v, alpha, beta


class Evd_lat_model(np_model):

    def __init__(self,
                 latent_encoder_output_size,
                 deterministic_encoder_output_size,
                 decoder_output_size,
                 args=None,
                 attention=None, ):
        super(Evd_lat_model, self).__init__(args)
        if args == None:
            raise NotImplementedError
        np_model.__init__(self, args)
        self.args = args

        if args.use_deterministic_path:  # CNP or ANP Model
            self._deterministic_encoder = ANPDeterministicEncoder(deterministic_encoder_output_size, attention)

        if args.use_latent_path:
            # self._latent_encoder = ANPLatentEncoder(latent_encoder_output_size)
            self._evidential_latent_encoder = ANPEvidentialLatentEncoder(latent_encoder_output_size, args=args)

        self._decoder = ANPDecoder(decoder_output_size, args=args)

    def forward(self, query, target_y=None, epoch=0):
        '''
        :param query: ( (context_x, context_y), target_x ).
        :param target_y:
        :param epoch:
        :return:
        '''
        (context_x, context_y), target_x = query


        # Skipped use latent path
        if self.args.use_latent_path:
            ##PRIOR
            ctx_lat_dist, ctx_nig_all, ctx_z_all = self._evidential_latent_encoder(context_x, context_y)

            # During training, we have target_y. We use the target for latent encoder

            if target_y is None:
                latent_rep_sample = ctx_lat_dist.rsample()
            else:
                ##POSTERIOR
                tar_lat_dist, tar_nig_all, tar_z_all = self._evidential_latent_encoder(target_x, target_y)
                latent_rep_sample = tar_lat_dist.rsample()

            batch_size, set_size, _ = target_x.shape
            latent_rep_sample = latent_rep_sample.unsqueeze(1).repeat([1, set_size, 1])

        if self._use_deterministic_path:
            deterministic_rep = self._deterministic_encoder(context_x, context_y, target_x)

        if self._use_deterministic_path and self._use_latent_path:
            representation = torch.cat((deterministic_rep, latent_rep_sample), dim=-1)
        elif self._use_latent_path:
            representation = latent_rep_sample
        elif self._use_deterministic_path:
            representation = deterministic_rep
        else:
            raise ValueError("You need at least one path for the encoder")

        dist, mu, sigma = self._decoder(representation, target_x)
        # print("mu: ", mu, "sigma: ", sigma)

        # Training tasks
        if target_y is not None:

            log_likelihood = dist.log_prob(target_y)

            loss_dict = {'Tr_NLL': float(-torch.mean(log_likelihood).detach().cpu().numpy())}

            # loss += torch.mean(0.5*(target_y-mu)**2)#-torch.mean(log_likelihood)
            loss = -torch.mean(log_likelihood)

            if self._use_latent_path:
                mu_lat_context = ctx_z_all[0]
                sigma_lat_context = ctx_z_all[1]

                mu_lat_tar = tar_z_all[0]
                sigma_lat_tar = tar_z_all[1]
                #
                gamma_c, v_c, alpha_c, beta_c = ctx_nig_all
                gamma_d, v_d, alpha_d, beta_d = tar_nig_all


                #kl_loss_dir = 0.5 * (torch.log(sigma_lat_context) - torch.log(
                #    sigma_lat_tar) + (sigma_lat_tar +
                #                      (mu_lat_tar - mu_lat_context) ** 2) / sigma_lat_context - 1)
                dist_1 = ctx_lat_dist #torch.distributions.Normal(mu_lat_context, torch.sqrt(sigma_sq_lat_context))
                dist_2 = tar_lat_dist #torch.distributions.Normal(mu_lat_tar, torch.sqrt(sigma_sq_lat_tar))
                kl_loss_dir = torch.distributions.kl_divergence(dist_2, dist_1)

                # kl_loss_norm = 0.5 * (torch.log(sigma_lat_context**2) - torch.log(
                #     sigma_lat_tar**2) + (sigma_lat_tar**2 +
                #                       (mu_lat_tar - mu_lat_context) ** 2) / sigma_lat_context**2 - 1)


                # kl_loss_nig =   0.5 * ( v_c * (gamma_c - gamma_d)**2 ) * alpha_d / beta_d \
                #               + 0.5 * v_c / v_d \
                #               - 0.5 * (torch.log(v_c/v_d)) \
                #               - 0.5 \
                #               + alpha_c * (torch.log(beta_d/beta_c)) \
                #               - torch.lgamma(alpha_d) + torch.lgamma(alpha_c) \
                #               + (alpha_d - alpha_c) * torch.digamma(alpha_d) \
                #               - (beta_d - beta_c) * alpha_d / beta_d

                kl_loss_nig = (2 * beta_c + v_c * (
                        gamma_c - gamma_d) ** 2 - 2 * beta_d) * alpha_d / 2 / beta_d + v_c / 2 / v_d - 1 / 2 + alpha_c * (
                                      torch.log(beta_d) - torch.log(beta_c)) + torch.lgamma(alpha_c) - torch.lgamma(
                    alpha_d) + (alpha_d - alpha_c) * torch.digamma(alpha_d) + (torch.log(v_d) - torch.log(v_c)) / 2

                # COnsider shape and think here
                loss_dict['Tr_KL_gaussian']=float(torch.mean(kl_loss_dir).detach().cpu().numpy())
                loss_dict['Tr_NIG_loss']=float(torch.mean(kl_loss_nig).detach().cpu().numpy())

                latent_path_loss = torch.mean(kl_loss_dir+kl_loss_nig)
                loss += latent_path_loss  # torch.mean(recons_loss + kl_loss)

                ctx_alea_lat = torch.mean( beta_c / (alpha_c - 1) )
                ctx_epis_lat = torch.mean( beta_c / (v_c * (alpha_c - 1)) )

                tar_alea_lat = torch.mean( beta_d / (alpha_d - 1) )
                tar_epis_lat = torch.mean( beta_d / (v_d * (alpha_d - 1)) )

                latent_uncertainties = ( ( float(ctx_epis_lat.detach().cpu().numpy()), float(ctx_alea_lat.detach().cpu().numpy()) ),
                                        ( float(tar_epis_lat.detach().cpu().numpy()), float(tar_alea_lat.detach().cpu().numpy()) ) )
                loss_dict['Tr_loss'] = float(loss.detach().cpu().numpy())
                return dist, loss, mu, sigma, latent_uncertainties, loss_dict


            else:
                print("Latent Evidential Model Needs Latent Path")
                raise NotImplementedError


        #Test Tasks
        else:


            if self.args.use_latent_path:
                gamma_c, v_c, alpha_c, beta_c = ctx_nig_all
                ctx_alea_lat = torch.mean(beta_c / (alpha_c - 1))
                ctx_epis_lat = torch.mean(beta_c / (v_c * (alpha_c - 1)))

                latent_uncertainties = ( ( float(ctx_epis_lat.detach().cpu().numpy()), float(ctx_alea_lat.detach().cpu().numpy()) ),
                                        ( float(ctx_epis_lat.detach().cpu().numpy()), float(ctx_alea_lat.detach().cpu().numpy()) ) )

                    #((ctx_epis_lat, ctx_alea_lat), (ctx_epis_lat, ctx_alea_lat))

                keys_tr = ['Tr_NLL', 'Tr_KL_gaussian', 'Tr_NIG_loss', 'Tr_loss']
                return dist, mu, sigma, latent_uncertainties, keys_tr


            else:
                print("The Latent Model")
                raise NotImplementedError


        print("Check settings.")
        raise ValueError

    def get_latent_representation(self, query):
        (context_x, context_y), target_x = query
        rep = self._evidential_latent_encoder.get_representation(context_x, context_y)
        return rep


    def for_active_forward(self, query):
        '''
        :param query: ( (context_x, context_y), target_x ).
        :param target_y:
        :param epoch:
        :return:
        '''
        (context_x, context_y), target_x = query

        # Skipped use latent path
        if self.args.use_latent_path:
            ##PRIOR
            ctx_lat_dist, ctx_nig_all, ctx_z_all = self._evidential_latent_encoder(context_x, context_y)

            gamma_c, v_c, alpha_c, beta_c = ctx_nig_all
            ctx_alea_lat = torch.mean(beta_c / (alpha_c - 1), dim = -1)

            ctx_epis_lat = torch.mean(beta_c / (v_c * (alpha_c - 1)), dim = -1)

            # context_uncertainties = torch.stack([ctx_epis_lat, ctx_alea_lat])
            # print("ctx unc: ", context_uncertainties)
            return ctx_epis_lat


        else:
            print("The Latent Model")
            raise NotImplementedError

    def test_forward_representation(self, query, target_y=None, epoch=0):
        '''
        :param query: ( (context_x, context_y), target_x ).
        :param target_y:
        :param epoch:
        :return:
        '''
        (context_x, context_y), target_x = query

        # Skipped use latent path
        if self._use_latent_path:
            ##PRIOR
            ctx_lat_dist, ctx_nig_all, ctx_z_all = self._evidential_latent_encoder(context_x, context_y)

            # During training, we have target_y. We use the target for latent encoder

            latent_rep_sample = ctx_lat_dist.rsample()

            batch_size, set_size, _ = target_x.shape
            latent_rep_sample = latent_rep_sample.unsqueeze(1).repeat([1, set_size, 1])

        if self._use_deterministic_path:
            deterministic_rep = self._deterministic_encoder(context_x, context_y, target_x)

        if self._use_deterministic_path and self._use_latent_path:
            representation = torch.cat((deterministic_rep, latent_rep_sample), dim=-1)
        elif self._use_latent_path:
            representation = latent_rep_sample
        elif self._use_deterministic_path:
            representation = deterministic_rep
        else:
            raise ValueError("You need at least one path for the encoder")

        representation = torch.mean(representation,dim=1)

        return representation


#############Attentive Models with self attention
from models.attention_model import *
class ANP_LatentModel(nn.Module):
    """
        Latent Model (Attentive Neural Process)
        Fixed Multihead Attention
        """

    def __init__(self, latent_encoder_sizes,
                 determministic_encoder_sizes,
                 decoder_output_size,
                 args,
                 attention,):
        super(ANP_LatentModel, self).__init__()
        num_hidden = latent_encoder_sizes[1]
        self.args = args

        self.latent_encoder = LatentEncoder(num_hidden, num_hidden,input_dim=latent_encoder_sizes[0])
        self.deterministic_encoder = DeterministicEncoder(num_hidden, num_hidden,input_dim=determministic_encoder_sizes[0])

        self.decoder = ANPDecoder(decoder_output_size, args=args)

    def forward(self,query, target_y=None):

        (context_x, context_y), target_x = query

        num_targets = target_x.size(1)

        if self.args.use_latent_path:
            ctx_lat_dist, ctx_lat_mu, ctx_lat_std = self.latent_encoder(context_x, context_y)

            if target_y is None:
                latent_rep_sample = ctx_lat_dist.rsample()
            else:
                tar_lat_dist, tar_lat_mu, tar_lat_std = self.latent_encoder(target_x, target_y)
                latent_rep_sample = tar_lat_dist.rsample()


        latent_rep_sample = latent_rep_sample.unsqueeze(1).repeat(1, num_targets, 1)  # [B, T_target, H]

        if self.args.use_deterministic_path:
            deterministic_rep = self.deterministic_encoder(context_x, context_y, target_x)  # [B, T_target, H]

        if self.args.use_deterministic_path and self.args.use_latent_path:
            representation = torch.cat((deterministic_rep, latent_rep_sample), dim=-1)
        elif self.args.use_latent_path:
            representation = latent_rep_sample
        elif self.args.use_deterministic_path:
            representation = deterministic_rep
        else:
            raise ValueError("You need at least one path for the encoder")

        dist, mu, std = self._decoder(representation, target_x)

        # For Training
        if target_y is not None:
            # get log probability
            log_likelihood = dist.log_prob(target_y)

            kl_loss = None
            loss = torch.zeros(size=(1,), device=target_y.device)

            # loss += torch.mean(0.5*(target_y-mu)**2)#-torch.mean(log_likelihood)
            recons_loss = -torch.mean(log_likelihood)
            loss += recons_loss

            # get KL divergence between prior and posterior
            if self.args.use_latent_path:
                dist_1 = torch.distributions.Normal(ctx_lat_mu, ctx_lat_std)
                dist_2 = torch.distributions.Normal(tar_lat_mu, tar_lat_std)
                kl_loss = torch.distributions.kl_divergence(dist_2, dist_1)

                loss += torch.mean(kl_loss)

        else:
            recons_loss = None
            kl_loss = None
            loss = None

        return dist, mu, std, recons_loss, kl_loss, loss



