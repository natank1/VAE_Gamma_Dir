import h5py
import torch
import data as data
import pathlib
import os.path as path
#This function was brought from here
# https://github.com/altosaar/variational-autoencoder/blob/master/train_variational_autoencoder_pytorch.py
def load_binary_mnist(cfg, **kwcfg):


        fname = cfg.data_dir + "fileall.hdf5"
        print (path.exists(fname))

        # # if not fname.exists():
        #     print('Downloading binary MNIST data...')
        #     data.download_binary_mnist(fname)
        #     print (data)
        #
        # f = h5py.File(pathlib.os.path.join(pathlib.os.environ['DAT'], 'binary_mnist.h5'), 'r')
        # exit(555)
        f = h5py.File(fname, 'r')
        x_train = f['train'][::]
        x_val = f['valid'][::]
        x_test = f['test'][::]
        train = torch.utils.data.TensorDataset(torch.from_numpy(x_train))
        train_loader = torch.utils.data.DataLoader(train, batch_size=cfg.batch_size, shuffle=True, **kwcfg)
        validation = torch.utils.data.TensorDataset(torch.from_numpy(x_val))
        val_loader = torch.utils.data.DataLoader(validation, batch_size=cfg.test_batch_size, shuffle=True)
        test = torch.utils.data.TensorDataset(torch.from_numpy(x_test))
        test_loader = torch.utils.data.DataLoader(test, batch_size=cfg.test_batch_size, shuffle=True)

        return train_loader, val_loader, test_loader





