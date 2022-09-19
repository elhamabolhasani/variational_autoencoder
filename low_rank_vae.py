import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, STL10
from torch.distributions.lowrank_multivariate_normal import LowRankMultivariateNormal
import os
import numpy as np
import pickle
import zipfile
import datetime
from shutil import copy


class EncoderModule(nn.Module):
    def __init__(self, input_channels, output_channels, stride, kernel, pad):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel, padding=pad, stride=stride)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Encoder(nn.Module):
    def __init__(self, color_channels, pooling_kernels, n_neurons_in_middle_layer):
        self.n_neurons_in_middle_layer = n_neurons_in_middle_layer
        super().__init__()
        self.bottle = EncoderModule(color_channels, 32, stride=1, kernel=1, pad=0)
        self.m1 = EncoderModule(32, 64, stride=1, kernel=3, pad=1)
        self.m2 = EncoderModule(64, 128, stride=pooling_kernels[0], kernel=3, pad=1)
        self.m3 = EncoderModule(128, 256, stride=pooling_kernels[1], kernel=3, pad=1)

    def forward(self, x):
        out = self.m3(self.m2(self.m1(self.bottle(x))))
        return out.view(-1, self.n_neurons_in_middle_layer)


class DecoderModule(nn.Module):
    def __init__(self, input_channels, output_channels, stride, activation="relu"):
        super().__init__()
        self.convt = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=stride, stride=stride)
        self.bn = nn.BatchNorm2d(output_channels)
        if activation == "relu":
            self.activation = nn.ReLU(inplace=False)  # set it true or false ??
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.bn(self.convt(x)))


class Decoder(nn.Module):
    def __init__(self, color_channels, pooling_kernels, decoder_input_size):
        self.decoder_input_size = decoder_input_size
        super().__init__()
        self.m1 = DecoderModule(256, 128, stride=1)
        self.m2 = DecoderModule(128, 64, stride=pooling_kernels[1])
        self.m3 = DecoderModule(64, 32, stride=pooling_kernels[0])
        self.bottle = DecoderModule(32, color_channels, stride=1, activation="sigmoid")

    def forward(self, x):
        out = x.view(-1, 256, self.decoder_input_size, self.decoder_input_size)
        out = self.m3(self.m2(self.m1(out)))
        return self.bottle(out)


class VAE(nn.Module):
    def __init__(self, dataset):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        assert dataset in ["mnist", "fashion-mnist", "cifar", "stl"]

        super().__init__()
        # # latent features
        self.n_latent_features = 64

        # resolution
        # mnist, fashion-mnist : 28 -> 14 -> 7
        # cifar : 32 -> 8 -> 4
        # stl : 96 -> 24 -> 6
        if dataset in ["mnist", "fashion-mnist"]:
            pooling_kernel = [2, 2]
            encoder_output_size = 7
        elif dataset == "cifar":
            pooling_kernel = [4, 2]
            encoder_output_size = 4
        elif dataset == "stl":
            pooling_kernel = [4, 4]
            encoder_output_size = 6

        # color channels
        if dataset in ["mnist", "fashion-mnist"]:
            color_channels = 1
        else:
            color_channels = 3

        # # neurons int middle layer
        n_neurons_middle_layer = 256 * encoder_output_size * encoder_output_size

        # Encoder
        self.encoder = Encoder(color_channels, pooling_kernel, n_neurons_middle_layer)
        # Middle
        self.fc1 = nn.Linear(n_neurons_middle_layer, self.n_latent_features)
        self.fc2 = nn.Linear(n_neurons_middle_layer, self.n_latent_features)
        self.fc3 = nn.Linear(n_neurons_middle_layer, self.n_latent_features)
        self.fc4 = nn.Linear(self.n_latent_features, n_neurons_middle_layer)
        # Decoder
        self.decoder = Decoder(color_channels, pooling_kernel, encoder_output_size)

        # data
        self.train_loader, self.test_loader = self.load_data(dataset)
        # history
        self.history = {"loss": [], "val_loss": []}

        # model name
        self.model_name = dataset + '_low_vae'
        if not os.path.exists(self.model_name):
            os.mkdir(self.model_name)

    def _reparameterize(self, mu, logvar, u):
        # sample from normal
        esp = torch.randn(*mu.size()).to(self.device)

        # std = logvar.exp_()
        std = torch.exp(logvar)
        u = u.view(-1, self.n_latent_features, 1)
        ut = u.view(-1, 1, self.n_latent_features)

        # make cov matirx
        std_mat = torch.diag_embed(std)
        ut_u = torch.matmul(u, ut)

        cov = std_mat + ut_u
        # change mean shape
        mu = mu.view(-1, self.n_latent_features, 1)

        # Set covariance function.
        K_0 = cov
        epsilon = 0.0001

        # Add small pertturbation.
        K = K_0 + torch.tensor(epsilon * np.identity(self.n_latent_features)).to(self.device)

        #  Cholesky decomposition.
        L = torch.linalg.cholesky(K)
        LL = torch.matmul(L, L.transpose(2, 1))

        z = mu + torch.matmul(L.float(), esp.view(-1, self.n_latent_features, 1))
        return z.view(-1, self.n_latent_features)

    def _bottleneck(self, h):
        mu, logvar, u = self.fc1(h), self.fc2(h), self.fc3(h)
        z = self._reparameterize(mu, logvar, u)
        return z, mu, logvar, u

    def sampling(self):
        # assume latent features space ~ N(0, 1)
        z = torch.randn(64, self.n_latent_features).to(self.device)
        z = self.fc4(z)
        # decode
        return self.decoder(z)

    def forward(self, x):
        # Encoder
        h = self.encoder(x)
        # Bottle-neck
        z, mu, logvar, u = self._bottleneck(h)
        # decoder
        z1 = self.fc4(z)
        d = self.decoder(z1)
        return d, mu, logvar, u

    # Data
    def load_data(self, dataset):
        data_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        if dataset == "mnist":
            train = MNIST(root="./data", train=True, transform=data_transform,
                          download=False)
            test = MNIST(root="./data", train=False, transform=data_transform,
                         download=False)
        elif dataset == "fashion-mnist":
            train = FashionMNIST(root="./data", train=True, transform=data_transform,
                                 download=True)
            test = FashionMNIST(root="./data", train=False, transform=data_transform,
                                download=True)
        elif dataset == "cifar":
            train = CIFAR10(root="./data", train=True, transform=data_transform,
                            download=False)
            test = CIFAR10(root="./data", train=False, transform=data_transform,
                           download=False)
        elif dataset == "stl":
            train = STL10(root="./data", split="unlabeled", transform=data_transform,
                          download=True)
            test = STL10(root="./data", split="test", transform=data_transform,
                         download=True)

        train_loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True, num_workers=0)
        test_loader = torch.utils.data.DataLoader(test, batch_size=64, shuffle=True, num_workers=0)

        return train_loader, test_loader

    def loss_function(self, recon_x, x, mu, logvar, u):
        uu = u.pow(2)  # tr{uut}
        # recounstruction error
        BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
        u = u.view(-1, self.n_latent_features, 1)
        ut = u.view(-1, 1, self.n_latent_features)
        std = torch.exp(logvar)  # .exp_()

        # log (|(1 + ut_std_inv_u)|)
        std_inv = 1 / std
        std_inv_mat = torch.diag_embed(std_inv)
        ut_std_inv = torch.matmul(ut, std_inv_mat)
        ut_std_inv_u = torch.matmul(ut_std_inv, u)
        det_cov = torch.log(torch.abs(1 + ut_std_inv_u))  # size(128, 1, 1)
        # + u.pow(2)+ log (|(1 + ut_std_inv_u)|)
        # logvar = log(|det(std_mat)|)
        KLD = -0.5 * (torch.sum(1 + logvar - mu.pow(2) - uu - logvar.exp()) + det_cov.sum())
        return BCE + KLD

    def init_model(self):
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

        if self.device == "cuda":
            self = self.cuda()
            torch.backends.cudnn.benchmark = True
        self.to(self.device)

    # Train
    def fit_train(self, epoch):
        self.train()
        print(f"\nEpoch: {epoch + 1:d} {datetime.datetime.now()}")
        train_loss = 0
        samples_cnt = 0
        for batch_idx, (inputs, _) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            self.optimizer.zero_grad()
            recon_batch, mu, logvar, u = self(inputs)

            loss = self.loss_function(recon_batch, inputs, mu, logvar, u)
            # loss.backward(retain_graph=True)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            samples_cnt += inputs.size(0)

            if batch_idx % 50 == 0:
                print(batch_idx, len(self.train_loader), f"Loss: {train_loss / samples_cnt:f}")

        self.history["loss"].append(train_loss / samples_cnt)

    def test(self, epoch):
        self.eval()
        val_loss = 0
        samples_cnt = 0
        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(self.test_loader):
                inputs = inputs.to(self.device)
                recon_batch, mu, logvar, u = self(inputs)
                val_loss += self.loss_function(recon_batch, inputs, mu, logvar, u).item()
                samples_cnt += inputs.size(0)

                if batch_idx == 0:
                    save_image(recon_batch, f"{self.model_name}/reconstruction_epoch_{str(epoch)}.png", nrow=8)

        print(batch_idx, len(self.test_loader), f"ValLoss: {val_loss / samples_cnt:f}")
        self.history["val_loss"].append(val_loss / samples_cnt)

        # sampling
        save_image(self.sampling(), f"{self.model_name}/sampling_epoch_{str(epoch)}.png", nrow=8)

    # save results
    def save_history(self, i):
        with open(f"{self.model_name}/{self.model_name}_history_{i}_.dat", "wb") as fp:
            pickle.dump(self.history, fp)

    def save_to_zip(self, i):
        with zipfile.ZipFile(f"{self.model_name}_{i}_.zip", "w") as zip:
            for file in os.listdir(self.model_name):
                zip.write(f"{self.model_name}/{file}", file)


train_phase = True
dataset_name = 'mnist'
model_path = 'checkpoints/low_vae_normal_latent_mnist_model_0201.pt'
optim_path = 'checkpoints/low_vae_normal_latent_mnist_optim_0201.pt'
if train_phase:
    net = VAE(dataset_name)
    net.init_model()
    for i in range(0, 500):
        net.fit_train(i)
        net.test(i)

        if i % 10 == 0 and i != 0:
            torch.save(
                net.state_dict(),
                "./checkpoints/low_vae_normal_latent_" + dataset_name + f"_model_{str(i + 1).zfill(4)}.pt"
            )
            torch.save(
                net.optimizer.state_dict(),
                "./checkpoints/low_vae_normal_latent_" + dataset_name + f"_optim_{str(i + 1).zfill(4)}.pt"
            )
            net.save_history(i + 1)
            net.save_to_zip(i + 1)

else:
    net = VAE(dataset_name)
    net.init_model()
    net.load_state_dict(torch.load(model_path))
    net.optimizer.load_state_dict(torch.load(optim_path))
    net.eval()

""" sampling part """


def generate_sample():
    z = torch.randn(1, net.n_latent_features).to(net.device)
    z = net.fc4(z)
    # decode
    return net.decoder(z)


directory = dataset_name + "_low_vae_generated_sample"
if not os.path.exists(directory):
    os.makedirs(directory)

for i in range(0, 10000):
    save_image(generate_sample(),
               dataset_name + "_low_vae_generated_sample/generared_image_" + str(i) + ".png", nrow=1)
