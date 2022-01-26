import torch.nn as nn
class np_model(nn.Module):
        def __init__(self,args):

            super(np_model, self).__init__()

            self._use_deterministic_path = args.use_deterministic_path
            self._use_latent_path = args.use_latent_path

        def forward(self):
            print("base template forward")

        def set_latent_path(self, value):
            self._use_latent_path = value

        def set_deterministic_path(self, value):
            self._use_deterministic_path = value


class ANPModel_template(np_model):

    def __init__(self,
                 latent_encoder_output_size,
                 deterministic_encoder_output_size,
                 decoder_output_size,
                 args = None,
                 attention=None, ):
        super(ANPModel_template, self).__init__(args)
        if args == None:
            raise NotImplementedError
        np_model.__init__(self, args)

        self._attention = attention
        self.latent_encoder_output_size = latent_encoder_output_size
        self.deterministic_encoder_output_size = deterministic_encoder_output_size
        self.decoder_output_size = decoder_output_size