from torchvision.utils import save_image
import torch
# import matplotlib.pyplot as plt
import numpy as np

from torch import nn
import yaml
import nomen
config = """
regular: True
simul: False
use_anal: True
use_metric: 2
use_greek: True
use_Dir: True

latent_size: 20
data_size: 784
learning_rate: 0.001
batch_size: 64
test_batch_size: 512
max_iterations: 100000
log_interval: 10000
early_stopping_interval: 5
n_samples: 600
use_gpu: False
train_dir: C:\\tt\\vae_trial\\
data_dir: C:\\tt\\vae_trial\\

image_folder: namefolde
model_dir: namefolde
rnd_mode: True
n_epochs: 6
seed: 582838
"""


def create_cfg():
    dictionary = yaml.load(config)
    cfg = nomen.Config(dictionary)
    cfg.parse_args()

    return cfg

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def generate_images(logits, cfg,nimages=5,imgg_w=28,img_h=28):

    yy = np.random.choice(logits.shape[1], nimages, replace=False)
    for j,i in enumerate(yy):
        save_image(logits[:64, i, :].view(cfg.batch_size, 1, imgg_w, img_h),
                   cfg.image_folder + "image_name+" + str(j)+"_"+str(np.random.randint(10000)) + "_.png")



class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        modules = [nn.Linear(input_size, hidden_size),
                   nn.ReLU(),
                   nn.Linear(hidden_size, hidden_size),
                   nn.ReLU(),
                   nn.Linear(hidden_size, output_size)]
        self.net = nn.Sequential(*modules)

    def forward(self, input):
        return self.net(input)


class NormalLogProb(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, loc, scale, z):
        var = torch.pow(scale, 2)
        return -0.5 * torch.log(2 * np.pi * var) - torch.pow(z - loc, 2) / (2 * var)


class BernoulliLogProb(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, target):
        # bernoulli log prob is equivalent to negative binary cross entropy
        return -self.bce_with_logits(logits, target)


def dump_model(model_name, encoder, decoder={}):
    states = { 'encoder': encoder.state_dict()}
    if not (decoder== {}):
        states.update({'decoder': decoder.state_dict()})

    torch.save(states, model_name )

#
# def prt_kl():
#     x = np.arange(-1.,1.,0.01)
#     y = np.arange(0.5,2.5,0.01)
#     x1,y1=np.meshgrid(x,y)
#     z= 0.5*(x1*x1-1+y1-np.log(y1))
#
#     fig = plt.figure()
#     ax = plt.axes(projection='3d')
#     ax.plot_surface(x1, y1, z, rstride=1, cstride=1,
#                 cmap='viridis', edgecolor='none')
#     ax.set_xlabel('mu')
#     ax.set_ylabel('sigma')
#     ax.set_zlabel('KL');
#     ax.set_title("KL on mu-sigma Plane")
#     plt.show()


if __name__ == "__main__":
    x =np.arange(80)
    yy =np.random.choice(80,4,replace=False)
    print(yy)
    for i in yy:
        print (i)
