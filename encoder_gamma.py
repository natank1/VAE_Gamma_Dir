import torch
from torch import nn
from metric_compare.utils import  NeuralNetwork,NormalLogProb
import metric_compare.utils as utils
import torch.distributions.dirichlet as d

from torch.distributions.gamma import  Gamma
import torch.distributions.kl as kl0

from metric_compare.get_mnist_data import load_binary_mnist


class EncoderGamma(nn.Module):
    """Approximate posterior parameterized by an inference network."""

    def __init__(self, cfg, target_function):
        super().__init__()
        self.inference_network = NeuralNetwork(input_size=cfg.data_size,
                                               output_size=2*cfg.latent_size ,
                                               hidden_size=2*cfg.latent_size )
        # Dirichlet function
        self.latent_size = cfg.latent_size
        self.batch_size =cfg.batch_size
        self.n_samples = cfg.n_samples
        self._gamm_alpha = torch.tensor(1.0).float()

        # self.alpha = self._gamm_alpha * torch.ones(self.batch_size,self.latent_size)
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


    # def get_z(self,x ):
    #     x = x.type(torch.FloatTensor)
    #     scale_arg = self.inference_network(x)
    #     scale = self.softplus(scale_arg)
    #     eps = torch.randn((loc.shape[0], self.n_samples, loc.shape[-1]))
    #     z = loc + scale * eps  # reparameterization
    #     return z,loc, scale

    def Dirich_KL(self,  scale):
        alpha0=torch.sum(scale,axis=2)
        
        kl_term_0 = torch.lgamma(alpha0) -torch.sum(torch.lgamma(scale)) -torch.lgamma(self.alpha0) + torch.sum(torch.lgamma(self.target_alpha))
        kl_term_1 = torch.digamma(scale) -self.digamma0
        kl_term_11= scale- self.target_alpha
        anal_kl = kl_term_0 + torch.mul(kl_term_1,kl_term_11)
        return anal_kl


    def create_generic_dir_stat(self):
        self.alpha0 = torch.sum(self.target_alpha)
        self.generic_mean = self.target_alpha / self.alpha0
        self.generic_dir = d.Dirichlet(self.target_alpha)
        self.creat_covariance()


    def forward(self, x):
        """Return sample of latent variable and log prob."""
        x = x.type(torch.FloatTensor)
        loc_arg, scale_arg = torch.chunk(self.inference_network(x), chunks=2, dim=-1)
        loc = self.softplus(loc_arg)
        scale = self.softplus(scale_arg)

        all_gama =Gamma(loc,scale)
        scores= kl0._kl_gamma_gamma(all_gama,self.target_function)
        scores = scores.mean(dim=-1)
        zrnd = all_gama.sample(sample_shape=(self.n_samples,))
        #find params

        mz = zrnd.mean(dim=0)
        ms = zrnd.var(dim=0)
        beta = mz / ms
        alpha = mz * beta
        gama_z=Gamma(alpha,beta)
        z_score=  kl0._kl_gamma_gamma(gama_z,self.target_function)
        z_score = z_score.mean(dim=-1)
        scores=torch.unsqueeze(scores,dim=-1)
        z_score = torch.unsqueeze(z_score, dim=-1)

        return zrnd, scores,z_score


if __name__  =='__main__':
    a=0
    cfg = utils.create_cfg()
    enc=EncoderGamma(cfg)
    print (enc)
    kwargs = {}
    train_data, _, _ = load_binary_mnist(cfg, **kwargs)

    xx =train_data.dataset[0:64,:]
    x1=xx[0]
    y = enc(x1)
    print (y.shape)
    exit(33)

    print (train_data.shape)
    # y= enc(x)
    # print (y.shape)
    m1=enc.generic_dir.sample()
    print (m1,torch.sum(m1),enc.generic_mean)
    print (enc.gen_cov)