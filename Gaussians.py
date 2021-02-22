import torch

#  This file contains set of functions for calcualting KL divergence for different scenarios  of Gaussians


from torch.distributions import multivariate_normal as gauss
from torch.distributions.multivariate_normal import _batch_mahalanobis

from torch.distributions import independent as indep

def kl_ind_mv(mvg0,indep0):
    delta_mean =mvg0.mean-indep0.mean
    indep_recip =torch.reciprocal(indep0.variance)
    mahlab_term= torch.sum(torch.mul(torch.mul(delta_mean,indep_recip),delta_mean),dim=-1)

    dimension = mvg0.event_shape[0]
    logdet_gauss = torch.logdet(mvg0.covariance_matrix)
    log_det_ind= torch.sum(torch.log(indep0.variance),dim=-1)
    partial_term =log_det_ind-logdet_gauss-dimension
    first_term =torch.sum(torch.diagonal(torch.mul(torch.unsqueeze(indep_recip,dim=-2),mvg0.covariance_matrix),dim1=-2,dim2=-1),dim=-1)
    kl_score =0.5*(partial_term+mahlab_term + first_term)
    return kl_score

def kl_ind_standard(indG):
    dimension = indG.event_shape[0]
    delta_2_norm = torch.pow(torch.norm(indG.mean, dim=-1), 2)
    log_det_ind = torch.sum(torch.log(indG.variance), dim=-1)
    trace_term =torch.sum(indG.variance,dim=-1)
    kl_score =0.5*(delta_2_norm+trace_term - log_det_ind-dimension)
    return kl_score
#
# #Univaraite case we simply receive two means and two standard deviations
#
#
# def kl_univ_gauss(mean1, sig1, mean2, sig2):
#     #sig is stnafard  dev no varaince !
#     kl_div=torch.log(sig2/sig1)+(torch.pow(sig1,2)+torch.pow(mean1-mean2,2))/(2*torch.pow(sig2,2)) -0.5
#     return kl_div
#
# # The batch case (nahdling batch broadcasting of matrices)
# def kl_mult_gauss_batch(mean1, cov1, mean2, cov2,dimension):
#
#     # cov2_det=torch.det(cov2)
#     # cov1_det = torch.det(cov1)
#     # log_ratio = torch.log(cov2_det / cov1_det)
#     log_ratio = torch.logdet(cov2) - torch.logdet(cov1)
#     inverse_2 =torch.inverse(cov2)
#
#     tmo_mat =torch.matmul(inverse_2,cov1)
#     tr_prod= torch.diagonal(tmo_mat, dim1=-2, dim2=-1).sum(-1)
#
#     delta_mean= torch.unsqueeze(mean1-mean2,dim=-1 )
#     aa= torch.matmul(inverse_2,delta_mean)
#     sq_prod= torch.squeeze(torch.squeeze(torch.matmul(torch.transpose(delta_mean,dim0=-2,dim1=-1),aa),dim=-1),dim=-1)
#
#     kl_div=0.5*(log_ratio-dimension+sq_prod +tr_prod)
#     return kl_div
#
#
# # The multivariate case (no batch) : mean1 adn mean2 are vectors (for the means ) and cov1 and cov2 are covraint matrix
# def kl_mult_gauss(mean1, cov1, mean2, cov2,dimension):
#
#
#
#     log_ratio = torch.logdet(cov2) - torch.logdet(cov1)
#     inverse_2 =torch.inverse(cov2)
#     tr_prod =torch.trace(torch.mm(inverse_2,cov1))
#
#     delta_mean= mean1-mean2
#     sq_prod= torch.matmul(delta_mean,torch.matmul(inverse_2,delta_mean))
#     kl_div=0.5*(log_ratio-dimension+sq_prod +tr_prod)
#     return kl_div
#
# # Here we assume that the covaraince matrices are diagonal hence held as vectors
# def kl_mult_gauss_diag(mean1, cov1, mean2, cov2,dimension,index=0):
#     log_ratio= torch.sum(torch.log(cov2),dim=-1) - torch.sum(torch.log(cov1),dim=-1)
#     recip_2 = torch.reciprocal(cov2)
#     delta_mean = mean1 - mean2
#     mat_prod= torch.sum(torch.mul(torch.mul(recip_2,delta_mean),delta_mean),dim=-1)
#     trace_like =torch.sum(torch.mul(recip_2,cov1),dim=-1)
#     kl_div = 0.5*(log_ratio-dimension+mat_prod+trace_like)
#     return kl_div
#
# def kl_mult_gauss_diag(diag_g0, diag_g1):
#
#     log_ratio= torch.sum(torch.log(diag_g1.scale),dim=-1) - torch.sum(torch.log(diag_g0.scale),dim=-1)
#     recip_2 = torch.reciprocal(diag_g1.scale)
#     delta_mean = diag_g0.loc - diag_g1.loc
#     mat_prod= torch.sum(torch.mul(torch.mul(recip_2,delta_mean),delta_mean),dim=-1)
#     trace_like =torch.sum(torch.mul(recip_2,diag_g0.scale),dim=-1)
#     kl_div = 0.5*(log_ratio-diag_g0.dimension+mat_prod+trace_like)
#     return kl_div
#
#
# # The notion standrrd assumes that we compare the matrix to the standrard Gaussian thus
# # we have obly a single gaussian
# def kl_mult_gauss_standard(mean1, cov1,dimension,index):
#
#
#     log_cov =torch.logdet(cov1)
#     tr_cov =torch.trace(cov1)
#     norm_mu = torch.pow(torch.norm(mean1,dim=index),2)
#     kl_div= 0.5 * (log_cov - dimension+ tr_cov + norm_mu)
#     return kl_div
#
#
# def kl_diag_standartd(mean1, cov1,dimension,index=0):
#
#     log_cov = -torch.sum(torch.log(cov1),dim=index)
#     tr_cov =torch.sum(cov1,dim=index)
#     norm_mu = torch.pow(torch.norm(mean1,dim=index),2)
#     kl_div= 0.5 * (log_cov - dimension+ tr_cov + norm_mu)
#     return kl_div
#
# def kl_new_diag_standartd(diag_g0):
#     log_cov = -torch.sum(torch.log(diag_g0.scale),dim=-1)
#     tr_cov =torch.sum(diag_g0.scale,dim=-1)
#     norm_mu = torch.pow(torch.norm(diag_g0.loc,dim=-1),2)
#     kl_div= 0.5 * (log_cov - diag_g0.dimension+ tr_cov + norm_mu)
#     return kl_div
#
