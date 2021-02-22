import  torch
import torch.utils
import torch.utils.data
from metric_compare.utils import NormalLogProb
import torch.distributions.kl as kl0
# import matplotlib.pyplot as plt
from metric_compare import utils as utils
from metric_compare.metric_backet import convert_metric_2_score
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.gamma import Gamma
def create_diriclet_samples(cfg,sample_size,dir_target,gamm_for_samp):
    overall_scres=[]

    for i in range(sample_size):
      zrnd=gamm_for_samp.sample(sample_shape =(5*sample_size,))
      mz =zrnd.mean(dim=0)


      create_d= Dirichlet(mz)
      score = kl0._kl_dirichlet_dirichlet(create_d,dir_target)
      score =convert_metric_2_score(torch.unsqueeze(score,dim=-1))

      overall_scres.append(score)


    return overall_scres

import matplotlib.pyplot as plt
if __name__ =='__main__':
    a=0
    cfg = utils.create_cfg()
    xx = torch.ones(cfg.batch_size, cfg.latent_size)
    yy = 2 * torch.ones(cfg.batch_size, cfg.latent_size)

    target_function = Gamma(xx, yy)
    rand_matrics= create_diriclet_samples(cfg,100,target_function)
    print (rand_matrics)
    plt.plot(rand_matrics[:10], 'r', label="Anal Encoder")
    plt.show()
    # plt.plot(rand_matrics, 'k', label="rand")
    #
    # plt.show()