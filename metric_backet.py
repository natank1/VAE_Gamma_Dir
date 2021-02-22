import torch



def kl_div_std_gauss_torch(mean_sample, var_sample):
    # Inouts are mean and varaice (not std! ,varaince)

    tt = mean_sample.mul(mean_sample)
    norm_mu = torch.sum(tt, axis=2)

    # norm_mu= mean_sample.matmul(mean_sample)
    cov_trace = torch.sum(var_sample, axis=2)
    log_cov = -torch.sum(torch.log(var_sample),axis=2)
    kl = 0.5 * (log_cov	- mean_sample.shape[2] + cov_trace+norm_mu)

    return kl



def dist_W2_torch(mean_sample, var_sample):
    # Inouts are mean and varaice (not std! ,varaince)
    lat_dim = mean_sample.shape[2]
    tt = mean_sample.mul(mean_sample)
    norm_mu = torch.sum(tt, axis=2)
    gen_cov =  (var_sample + torch.ones(lat_dim) - 2 * torch.sqrt(var_sample))
    gen_trace = torch.sum(gen_cov,axis=2)
    score =norm_mu + gen_trace
    return score

def convert_metric_2_score (kl_score):
    loss = kl_score.mean(1).sum(0)
    real_score = loss.detach().numpy()
    return real_score,  loss


if __name__ =='__main__':
    a=0
    x=torch.tensor([1.2,2.,0.5])
    x1=x+torch.ones(3)-2*torch.sqrt(x)
    print (x1)
    tt=torch.eye(3)*(x+torch.ones(3)-2*torch.sqrt(x))
    print(tt)

    print (tt.trace())
    exit(44)
    y= torch.tensor([3,1,0.1])