import  torch
import torch.utils
import torch.utils.data
from metric_compare.utils import NormalLogProb
import torch.distributions.kl as kl0
# import matplotlib.pyplot as plt
from metric_compare import utils as utils
from metric_compare.metric_backet import convert_metric_2_score
from torch.distributions.gamma import  Gamma

def create_Gamma_samples(cfg,sample_size,Gamma_target):
    overall_scres=[]

    for i in range(sample_size):
      zrnd=Gamma_target.sample(sample_shape =(10*sample_size,))
      mz =zrnd.mean(dim=0)
      ms =zrnd.var(dim=0)
      beta =mz/ms
      alpha =mz*beta

      create_g= Gamma(alpha,beta)
      score = kl0._kl_gamma_gamma(create_g,Gamma_target)
      score =convert_metric_2_score(torch.unsqueeze(score,dim=-1))

      overall_scres.append(score)

      # overall_scres.append(score.detach().numpy()[0])
      # print (overall_scres)
    return overall_scres

import matplotlib.pyplot as plt
if __name__ =='__main__':
    a=0
    cfg = utils.create_cfg()
    xx = torch.ones(cfg.batch_size, cfg.latent_size)
    yy = 2 * torch.ones(cfg.batch_size, cfg.latent_size)

    target_function = Gamma(xx, yy)
    rand_matrics= create_Gamma_samples(cfg,100,target_function)
    print (rand_matrics)
    plt.plot(rand_matrics[:10], 'r', label="Anal Encoder")
    plt.show()
    # plt.plot(rand_matrics, 'k', label="rand")
    #
    # plt.show()