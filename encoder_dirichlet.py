import torch
from torch import nn
from metric_compare.utils import  NeuralNetwork,NormalLogProb

from torch.distributions.dirichlet import Dirichlet
from torch.distributions.gamma import  Gamma
import torch.distributions.kl as kl0


class EncoderDir(nn.Module):
    """Approximate posterior parameterized by an inference network."""

    def __init__(self, cfg, target_function):
        super().__init__()
        self.inference_network = NeuralNetwork(input_size=cfg.data_size,
                                               output_size=cfg.latent_size ,
                                               hidden_size=cfg.latent_size )
        # Dirichlet function
        self.latent_size = cfg.latent_size
        self.batch_size =cfg.batch_size
        self.n_samples = cfg.n_samples
        self._gamm_alpha = torch.tensor(1.0).float()

        self.all_beta = torch.ones(self.batch_size,self.latent_size)
         #Constrcuting the referene

        self.target_function= target_function
        # self.create_generic_dir_stat()


        # self.digamma0 =torch.digamma(self.target_alpha)
        self.log_p= NormalLogProb()
        self.cfg=cfg
        self.softplus = nn.Softplus()
        self.register_buffer('p_z_loc', torch.zeros(cfg.latent_size))
        self.register_buffer('p_z_scale', torch.ones(cfg.latent_size))

        self.encoder_score = self.Dirich_KL




    def Dirich_KL(self,  scale):
        alpha0=torch.sum(scale,axis=2)
        
        kl_term_0 = torch.lgamma(alpha0) -torch.sum(torch.lgamma(scale)) -torch.lgamma(self.alpha0) + torch.sum(torch.lgamma(self.target_alpha))
        kl_term_1 = torch.digamma(scale) -self.digamma0
        kl_term_11= scale- self.target_alpha
        anal_kl = kl_term_0 + torch.mul(kl_term_1,kl_term_11)
        return anal_kl





    def forward(self, x):
        """Return sample of latent variable and log prob."""
        x = x.type(torch.FloatTensor)
        scale_arg = self.inference_network(x)
        scale = self.softplus(scale_arg)

        all_gama =Gamma(scale,self.all_beta)
        all_dir= Dirichlet(scale)
        scores= kl0._kl_dirichlet_dirichlet(all_dir,self.target_function)
        scores = scores.mean(dim=-1)

        zrnd = all_gama.sample(sample_shape=(self.n_samples,))
        #find params

        mz = zrnd.mean(dim=0)
        z_dir = Dirichlet(mz)
        z_score=  kl0._kl_dirichlet_dirichlet(z_dir,self.target_function)
        z_score = z_score.mean(dim=-1)
        scores=torch.unsqueeze(scores,dim=-1)
        z_score = torch.unsqueeze(z_score, dim=-1)

        return zrnd, scores,z_score

