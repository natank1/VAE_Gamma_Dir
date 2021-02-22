import  torch
import torch.utils
import torch.utils.data
import random
import numpy as np
import metric_compare.utils as utils
from metric_compare.metric_backet import convert_metric_2_score
from metric_compare.generate_random_sample import create_kl_gaussian_samples
from metric_compare.generate_gamma_sample import  create_Gamma_samples
from metric_compare.generate_dirichlet_sample import  create_diriclet_samples
from metric_compare.get_mnist_data import load_binary_mnist
from metric_compare.encoder_mean_field import EncoderMF
from metric_compare.encoder_gamma import EncoderGamma
from metric_compare.encoder_dirichlet import EncoderDir
from torch.distributions.gamma import  Gamma
from torch.distributions.dirichlet import Dirichlet


def train_only_enc(cfg,encoder, train_data ):


    optimizer1 = torch.optim.RMSprop(list(encoder.parameters()),
                                     lr=cfg.learning_rate,
                                     centered=True)

    kl_scores=[]
    mc_scores=[]
    mc_scores2 = []
    for epoc in range(cfg.n_epochs):
        best_valid_elbo = -np.inf
        cntr=0
        for step, batch in enumerate(utils.cycle(train_data)):
            x = batch[0].to("cpu")
            cntr+=1

            encoder.zero_grad()
            z, enc_score,score2 = encoder(x)

            #Only to be alligned with previous version it is redundant
            enc_score= torch.unsqueeze(enc_score,dim=-1)
            score2= torch.unsqueeze(score2,dim=-1)

            # average over sample dimension
            optimizer1.zero_grad()
            real_score, loss= convert_metric_2_score(enc_score)
            real_score2, _ = convert_metric_2_score(score2)

            # b0=create_torch_samples(cfg)
            if torch.isnan(z)[0, 0, 0] == torch.tensor(False):
               loss.backward()
               optimizer1.step()
               kl_scores.append(real_score)
               mc_scores.append((real_score2))

            if cntr >600:
               break

    return kl_scores, mc_scores
    # print ("final results")
    # with torch.no_grad():
    #     _ = evaluate_enc(cfg.n_samples, decoder, encoder, eval_data, combined=False)

import matplotlib.pyplot as plt
if __name__ == '__main__':

    sample_size_to_compare =100
    cfg =utils.create_cfg()

    if cfg.use_greek:
         if cfg.use_Dir:
             xx = torch.ones(cfg.batch_size, cfg.latent_size)

             our_tensor =torch.randint(1,6, (1, cfg.latent_size))
             our_tensor =our_tensor.repeat(cfg.batch_size,1)
             our_tensor =our_tensor.float()
             Gamm_forsamp = Gamma(our_tensor,xx)

             target_function = Dirichlet(our_tensor)
             rand_matrics = create_diriclet_samples(cfg, sample_size_to_compare, target_function,Gamm_forsamp)
             print (rand_matrics)

             encoder = EncoderDir(cfg, target_function)
         else:
            xx = 5*torch.ones(cfg.batch_size, cfg.latent_size)
            yy =  2*torch.ones(cfg.batch_size, cfg.latent_size)
            target_function = Gamma(xx, yy)
            rand_matrics = create_Gamma_samples(cfg, sample_size_to_compare, target_function)
            encoder = EncoderGamma(cfg, target_function)
    else :
        rand_matrics = create_kl_gaussian_samples(cfg, sample_size_to_compare)
        encoder = EncoderMF(cfg)



    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    # decoder = Decoder(latent_size=cfg.latent_size, data_size=cfg.data_size)
    device="cpu"
    # decoder.to(device)


    encoder.to(device)
    kwargs = {}

    train_data, _, _ = load_binary_mnist(cfg, **kwargs)

    kl_scores,mc_scores= train_only_enc(cfg, encoder,  train_data)
    kl_scores0 = kl_scores[:sample_size_to_compare]
    mc_scores0 = mc_scores[:sample_size_to_compare]

    kl_scores1=kl_scores[-sample_size_to_compare:]
    mc_scores1 = mc_scores[-sample_size_to_compare:]
    # print (kl_scores0[:10])
    # print(kl_scores1)
    # print(mc_scores1)
    # print (rand_matrics)
    #
    # plt.plot(kl_scores0[:10], 'r', label="Anal Encoder")
    #
    # plt.plot(rand_matrics[:10], 'k', label="rand")
    # plt.plot(mc_scores0[:10], 'g', label="MC Est")
    #
    # plt.title("Early MC EstTraining")
    # plt.legend()
    # plt.show()
    print(rand_matrics)
    plt.plot(kl_scores1, 'r', label="Anal Encoder")
    plt.plot(rand_matrics, 'k', label="rand")
    plt.plot(mc_scores1, 'g', label="Z's")

    plt.legend()
    plt.title("Dirichlet Training")

    plt.show()


    plt.plot(kl_scores1, 'r', label="Anal Encoder")


    plt.title("Dirichlet  Encoder Parameters")
    plt.legend()
    plt.show()
