
import torch
import scipy.linalg as lin_alg
import numpy as np
from torch.distributions import multivariate_normal as gauss
import torch.distributions.kl
# The formaula is taken from here

# https://djalil.chafai.net/blog/2010/04/30/wasserstein-distance-between-two-gaussians/

# Taking the fractional power of non diagonal covaraicne
def covar_frac_power(covar_mat):
    eigen_val, eig_vec = torch.symeig(covar_mat, eigenvectors=True)
    sqrt_eig = torch.diag_embed(torch.sqrt(eigen_val), offset=0, dim1=-2, dim2=-1)
    vec_transp = eig_vec.transpose(-2, -1)
    sqrt_cov1 = torch.matmul(torch.matmul(eig_vec, sqrt_eig), vec_transp)
    return sqrt_cov1

#Norm two of two vector means of Guassians
def norm_p_of_mean_differences(mean1, mean2,p=2):
    delta_mean =mean1 - mean2
    delta_2_norm = torch.pow(torch.norm(delta_mean, dim=-1), p)
    return delta_2_norm
# This function calcuates the wasssertein 2 of two gaussians with non diagonal covaraicne matrices
def dist_W2_mv_mv(mv_gauss0, mv_gauss1 ):

    delta_2_norm=norm_p_of_mean_differences(mv_gauss0.mean, mv_gauss1.mean)

    sqrt_cov1= covar_frac_power(mv_gauss0.covariance_matrix)
    mat_prod =torch.matmul(torch.matmul(sqrt_cov1,mv_gauss1.covariance_matrix),sqrt_cov1)
    sqrt_mat2= covar_frac_power(mat_prod)

    sumdi= 2*torch.diagonal(sqrt_mat2,dim1=-2,dim2=-1).sum(-1)

    cov_trace= torch.diagonal(mv_gauss0.covariance_matrix+mv_gauss1.covariance_matrix,dim1=-2,dim2=-1).sum(-1)

    w2 =delta_2_norm+cov_trace-sumdi
    return w2

#Wassertein metric for diaganon gaussian with multivariate Gaussian (name ly diagonal covarraince with non diagonal )
def dist_W2_indepen_mv(diag_gauss,mv_gauss ):
    delta_2_norm = norm_p_of_mean_differences(diag_gauss.mean, mv_gauss.mean)
    conv_dimn=torch.unsqueeze(diag_gauss.stddev,dim=-2)
    mat_prod =torch.mul(torch.mul(conv_dimn,mv_gauss.covariance_matrix),conv_dimn)
    sqrt_mm = covar_frac_power(mat_prod)
    sumdi= 2*torch.diagonal(sqrt_mm,dim1=-2,dim2=-1).sum(-1)
    diag_tracce = torch.sum(diag_gauss.variance,dim=-1)
    mv_trace=torch.diagonal(mv_gauss.covariance_matrix,dim1=-2,dim2=-1).sum(-1)
    w2 =delta_2_norm+diag_tracce +mv_trace-sumdi
    return w2



#Wassertein2 Between diagoanl covaraince matrix and the Standard normal matrix (one to be used in problems such as VAE)
def dist_W2_indepdnet_sn(independent_gauss ):
    norm_mu = torch.pow(torch.norm(independent_gauss.mean,dim=-1),2)
    mega_trace=2*torch.sum(independent_gauss.stddev,dim=-1)
    tr1 =torch.sum(independent_gauss.variance,dim=-1)
    w2= norm_mu+tr1+independent_gauss.event_shape[0] -mega_trace

    return w2

# Wassertein2 for two indepdnedt Gaussians (diagonal covaaince)
def dist_W2_ind_ind(independent_gauss_0, independent_gauss_1):
    delta_2_norm = norm_p_of_mean_differences(independent_gauss_0.mean, independent_gauss_1.mean)
    sqrt_cov1 = independent_gauss_0.stddev
    mega_mat=  independent_gauss_0.variance+independent_gauss_1.variance-2*torch.sqrt(torch.mul(sqrt_cov1,torch.mul(independent_gauss_1.variance,sqrt_cov1)))
    w2= delta_2_norm+torch.sum(mega_mat,dim=-1)
    return w2


# Kl divergence  unique cases

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
if __name__ =='__main__':
   from torch.distributions import independent
   from torch.distributions import multivariate_normal,normal
   loc = torch.rand(3)
   scale= 2+torch.rand(3)
   xx= multivariate_normal.MultivariateNormal(loc,torch.eye(3)*scale)
   xy = multivariate_normal.MultivariateNormal(torch.zeros(3),torch.eye(3) )
   print (loc)
   print (scale)
   d1 = independent.Independent(normal.Normal(loc,scale),1)
   xz=multivariate_normal.MultivariateNormal(loc,torch.eye(3)*torch.pow(scale,2) )
   loc = torch.rand(3)
   scale = 2 + torch.rand(3)
   print(loc)
   print(scale)
   d2 = independent.Independent(normal.Normal(loc, scale), 1)
   d3 = independent.Independent(normal.Normal(torch.zeros(3),torch.ones(3)), 1)

   print (dist_W2_ind_ind(d1, d2))
   print(dist_W2_ind_ind(d1, d3))
   print ("tt ",dist_W2_indepen_mv(d1, xy))
   print("dd ",dist_W2_mv_mv(xz, xy))
   print(dist_W2_indepdnet_sn(d1))


