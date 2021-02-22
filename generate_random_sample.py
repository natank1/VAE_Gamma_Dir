import  torch
import torch.utils
import torch.utils.data
from metric_compare.metric_backet import  kl_div_std_gauss_torch, convert_metric_2_score,dist_W2_torch
from metric_compare import utils as utils
import metric_compare.Metrics.Wasserstein_metric_for_Gaussians2 as wass
import metric_compare.Metrics.Gaussians as gauss
 
from torch.distributions import independent
from torch.distributions import multivariate_normal,normal

def create_kl_gaussian_samples(cfg,sample_size):
    overall_scres=[]
    if cfg.use_anal:
       if cfg.use_metric==1:
           proc_calc = gauss.kl_ind_standard

       if cfg.use_metric == 2:

           proc_calc = wass.dist_W2_indepdnet_sn

           # proc_calc = wass.dist_W2_diag

    for i in range(sample_size):
      zrnd=torch.randn(cfg.batch_size,cfg.n_samples,cfg.latent_size)


      loc = torch.mean(zrnd, axis=1).unsqueeze(axis=1)
      scale = torch.std(zrnd, axis=1).unsqueeze(axis=1)
      idn_struc= independent.Independent(normal.Normal(loc, scale), 1)
      kl = proc_calc(idn_struc)
      real_score, _ =convert_metric_2_score(kl)
      overall_scres.append(real_score)

      # overall_scres.append(score.detach().numpy()[0])
      # print (overall_scres)

    return overall_scres

# def create_Gamma_samples(cfg,sample_size):
#     overall_scres=[]
#     if cfg.use_anal:
#        if cfg.use_metric==1:
#            proc_calc = kl_div_std_gauss_torch
#            proc_calc = gauss.kl_new_diag_standartd
#
#        if cfg.use_metric == 2:
#            proc_calc = dist_W2_torch
#     for i in range(sample_size):
#       zrnd=torch.randn(cfg.batch_size,cfg.n_samples,cfg.latent_size)
#
#       loc = torch.mean(zrnd, axis=1).unsqueeze(axis=1)
#       scale = torch.var(zrnd, axis=1).unsqueeze(axis=1)
#       # kl= proc_calc(loc, torch.unsqueeze(torch.eye(cfg.latent_size) * scale, dim=1),
#       #                    torch.zeros(size=(cfg.batch_size, 1, cfg.latent_size)), dd, 2)
#
#       kl = proc_calc(loc, torch.unsqueeze(torch.eye(cfg.latent_size) * scale, dim=1),
#                      torch.zeros(size=(cfg.batch_size, 1, cfg.latent_size)), dd, 2)
#
#       real_score, _ =convert_metric_2_score(kl)
#       overall_scres.append(real_score)
#
#       # overall_scres.append(score.detach().numpy()[0])
#       # print (overall_scres)
#     return overall_scres
#

if __name__ =='__main__':
    a=0
    cfg = utils.create_cfg()
    rand_matrics= create_kl_gaussian_samples(cfg,100)
    # plt.plot(rand_matrics, 'k', label="rand")
    #
    # plt.show()