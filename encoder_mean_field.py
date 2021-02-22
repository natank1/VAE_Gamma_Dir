import torch
from torch import nn
from metric_compare.utils import  NeuralNetwork,NormalLogProb

import  metric_compare.Metrics.Gaussians as gauss
import  metric_compare.Metrics.Wasserstein_metric_for_Gaussians2 as wass
from  metric_compare.Metrics.multivarGaussdiag import diagGauss
from torch.distributions import independent
from torch.distributions import normal
from torch.distributions import multivariate_normal
class EncoderMF(nn.Module):
    """Approximate posterior parameterized by an inference network."""

    def __init__(self, cfg):
        super().__init__()
        self.inference_network = NeuralNetwork(input_size=cfg.data_size,
                                               output_size=cfg.latent_size * 2,
                                               hidden_size=cfg.latent_size * 2)


        self.log_p= NormalLogProb()
        self.latent_size =cfg.latent_size
        self.batch_size= cfg.batch_size
        self.cfg=cfg
        self.softplus = nn.Softplus()
        self.register_buffer('p_z_loc', torch.zeros(cfg.latent_size))
        self.register_buffer('p_z_scale', torch.ones(cfg.latent_size))
        self.n_samples = cfg.n_samples
        if cfg.use_metric==1:
             self.encoder_score = self.analytic_score
        elif cfg.use_metric == 2:
             self.encoder_score = self.analytic_w2_score


    def get_z(self,x ):
        x = x.type(torch.FloatTensor)
        loc, scale_arg = torch.chunk(self.inference_network(x).unsqueeze(1), chunks=2, dim=-1)
        scale = self.softplus(scale_arg)
        eps = torch.randn((loc.shape[0], self.n_samples, loc.shape[-1]))
        z = loc + scale * eps  # reparameterization
        return z,loc, scale



    def analytic_score(self, loc, scale, z):
        # scale = scale.pow(2)  #conver std to var


        d1=independent.Independent(normal.Normal(torch.squeeze(loc), torch.squeeze(scale)), 1)
        anal_kl = gauss.kl_ind_standard(d1)


        return anal_kl

    def analytic_w2_score(self, loc, scale, z):
        # scale = scale.pow(2)  #conver std to var

        d1 = independent.Independent(normal.Normal(loc, scale), 1)
        if torch.isnan(scale)[0,0,0]==torch.tensor((True)) :
         return torch.zeros(self.batch_size,1)
        anal_kl = wass.dist_W2_indepdnet_sn(d1)

        return anal_kl

    def forward(self, x):
        """Return sample of latent variable and log prob."""

        z,loc,scale =self.get_z(x )
        # tt=torch.distributions.Normal(loc, scale).sample(sample_shape=(80,))
        # zrnd = torch.randn(cfg.batch_size, cfg.n_samples, cfg.latent_size)
        loc2=z.mean(axis=1,keepdim=True)
        scale2=z.std(axis=1,keepdim=True)
        score_enc= self.encoder_score(loc,scale,z)
        score2 =self.encoder_score(loc2, scale2,z)
        return z, score_enc,score2