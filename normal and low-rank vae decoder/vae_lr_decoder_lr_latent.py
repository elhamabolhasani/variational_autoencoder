import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, STL10
import os
import pickle
import zipfile
import datetime
from shutil import copy
from torch.distributions import Normal, LowRankMultivariateNormal
import numpy as np


class EncoderModule(nn.Module):
    def __init__(self, input_channels, output_channels, stride, kernel, pad):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel, padding=pad, stride=stride)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

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
            self.activation = nn.ReLU(inplace=True)
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
        # self.bottle = DecoderModule(32, color_channels, stride=1, activation="sigmoid")**
        self.m4 = DecoderModule(32, color_channels, stride=1)

    def forward(self, x):
        out = x.view(-1, 256, self.decoder_input_size, self.decoder_input_size)
        out = self.m3(self.m2(self.m1(out)))
        return self.m4(out)  # **


class VAE(nn.Module):
    def __init__(self, dataset):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(self.device)
        assert dataset in ["mnist", "fashion-mnist", "cifar", "stl"]

        super().__init__()
        # # latent features
        self.n_latent_features = 256   # TODO ---> 128, 256, 2048,

        # resolution
        # mnist, fashion-mnist : 28 -> 14 -> 7
        # cifar : 32 -> 8 -> 4
        # stl : 96 -> 24 -> 6
        if dataset in ["mnist", "fashion-mnist"]:
            pooling_kernel = [2, 2]
            encoder_output_size = 7
            self.decoder_output_size = 28
        elif dataset == "cifar":
            pooling_kernel = [4, 2]
            encoder_output_size = 4
            self.decoder_output_size = 32
        elif dataset == "stl":
            pooling_kernel = [4, 4]
            encoder_output_size = 6

        # color channels
        if dataset in ["mnist", "fashion-mnist"]:
            self.color_channels = 1
        else:
            self.color_channels = 3

        # # neurons int middle layer
        n_neurons_middle_layer = 256 * encoder_output_size * encoder_output_size
        self.n_neurons_last_decoder_layer = self.color_channels * self.decoder_output_size * self.decoder_output_size
        print(self.n_neurons_last_decoder_layer)
        # Encoder
        self.encoder = Encoder(self.color_channels, pooling_kernel, n_neurons_middle_layer)
        # Middle
        self.fc1 = nn.Linear(n_neurons_middle_layer, self.n_latent_features)
        self.fc2 = nn.Linear(n_neurons_middle_layer, self.n_latent_features)
        self.fc3 = nn.Linear(n_neurons_middle_layer, self.n_latent_features)
        self.fc4 = nn.Linear(self.n_latent_features, n_neurons_middle_layer)
        # Decoder
        self.decoder = Decoder(self.color_channels, pooling_kernel, encoder_output_size)
        self.fc5 = nn.Linear(self.n_neurons_last_decoder_layer,
                             self.decoder_output_size * self.decoder_output_size * self.color_channels)  # check if its true ?
        self.fc6 = nn.Linear(self.n_neurons_last_decoder_layer,
                             self.decoder_output_size * self.decoder_output_size * self.color_channels)
        self.fc7 = nn.Linear(self.n_neurons_last_decoder_layer,
                             self.decoder_output_size * self.decoder_output_size * self.color_channels)

        # data
        self.train_loader, self.test_loader = self.load_data(dataset)
        # history
        self.history = {"loss": [], "val_loss": []}

        # model name
        self.model_name = dataset + '_vae_lr_decoder_lr_latent_test'
        if not os.path.exists(self.model_name):
            os.mkdir(self.model_name)

        self.log_norm_constant = -0.5 * np.log(2 * np.pi)

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

    def decoder_bottleneck(self, d):
        mu, logvar, u = self.fc5(d), self.fc6(d), self.fc7(d)
        return mu, logvar, u

    def sampling(self):
        # assume latent features space ~ N(0, 1)
        z = torch.randn(64, self.n_latent_features).to(self.device)
        z = self.fc4(z)
        z = self.decoder(z)
        z = z.view(-1, self.n_neurons_last_decoder_layer)
        mu_d, logvar_d, u_d = self.decoder_bottleneck(z)

        sigma = torch.exp(logvar_d) + 0.01

        dist = LowRankMultivariateNormal(mu_d, u_d.view(-1, u_d.shape[1], 1), sigma)

        # sample from model
        z = dist.sample()
        return z.view(-1, self.color_channels, self.decoder_output_size, self.decoder_output_size)

    def forward(self, x):
        # Encoder
        h = self.encoder(x)
        # Bottle-neck
        z, mu, logvar, u = self._bottleneck(h)
        # decoder
        z = self.fc4(z)
        d = self.decoder(z)
        d_ = d.view(-1, self.n_neurons_last_decoder_layer)
        mu_d, logvar_d, u_d = self.decoder_bottleneck(d_)
        return mu_d, logvar_d, u_d, mu, logvar, u

    # Data
    def load_data(self, dataset):
        data_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        if dataset == "mnist":
            train = MNIST(root="./data", train=True, transform=data_transform, download=False)
            test = MNIST(root="./data", train=False, transform=data_transform, download=False)
        elif dataset == "fashion-mnist":
            train = FashionMNIST(root="./data", train=True, transform=data_transform,
                                 download=True)
            test = FashionMNIST(root="./data", train=False, transform=data_transform,
                                download=True)
        elif dataset == "cifar":
            train = CIFAR10(root="./data", train=True, transform=data_transform,
                            download=True)
            test = CIFAR10(root="./data", train=False, transform=data_transform,
                           download=True)
        elif dataset == "stl":
            train = STL10(root="./data", split="unlabeled", transform=data_transform,
                          download=True)
            test = STL10(root="./data", split="test", transform=data_transform, download=True)
        if dataset == "mnist":
            train_loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True, num_workers=0)
            test_loader = torch.utils.data.DataLoader(test, batch_size=64, shuffle=True, num_workers=0)
        else:
            train_loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True, num_workers=0)
            test_loader = torch.utils.data.DataLoader(test, batch_size=64, shuffle=True, num_workers=0)

        return train_loader, test_loader

    def loss_function(self, mu_d, logvar_d, u_d, x, mu, logvar, u):
        # BCD
        sigma = torch.exp(logvar_d) + 0.01
        dist = LowRankMultivariateNormal(mu_d, u_d.view(-1, u_d.shape[1], 1), sigma)
        d = dist.log_prob(x.view(-1, x.shape[2] * x.shape[2] * self.color_channels))
        # print("BCE : ", d.sum())
        # print("KLD : ", KLD)
        # print("BCE + KLD : ", d.sum() + KLD)

        # sample from model
        z = dist.sample()

        # KLD
        uu = u.pow(2)  # tr{uut}
        # recounstruction error
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

        return KLD - d.sum(), z.view(-1, self.color_channels, x.shape[2], x.shape[2])

    def init_model(self):
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)

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
            mu_d, logvar_d, u_d, mu, logvar, u = self(inputs)
            loss, z = self.loss_function(mu_d, logvar_d, u_d, inputs, mu, logvar, u)
            # print("loss item :" , loss.item())
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
                mu_d, logvar_d, u_d, mu, logvar, u = self(inputs)
                loss, z = self.loss_function(mu_d, logvar_d, u_d, inputs, mu, logvar, u)
                val_loss += loss.item()
                samples_cnt += inputs.size(0)

                if batch_idx == 0:
                    save_image(z, f"{self.model_name}/reconstruction_epoch_{str(epoch)}.png", nrow=8)

        print(batch_idx, len(self.test_loader), f"ValLoss: {val_loss / samples_cnt:f}")
        self.history["val_loss"].append(val_loss / samples_cnt)

        # sampling
        print("saved image  ", f"{self.model_name}/sampling_epoch_{str(epoch)}.png")

        save_image(self.sampling(), f"{self.model_name}/sampling_epoch_{str(epoch)}.png", nrow=8)

    # save results
    def save_history(self, i):
        with open(f"{self.model_name}/{self.model_name}_history_{i}_.dat", "wb") as fp:
            pickle.dump(self.history, fp)

    def save_to_zip(self, i):
        with zipfile.ZipFile(f"{self.model_name}_{i}_.zip", "w") as zip:
            for file in os.listdir(self.model_name):
                zip.write(f"{self.model_name}/{file}", file)

        # copy(f"{self.model_name}_{i}_.zip", "/content/drive/MyDrive/VAE_Project/model/reform low vae model/")


train_phase = True
model_path = '/checkpoints/vae_lr_decoder_lr_latent_cifar_model.pt'
optim_path = '/checkpoints/vae_lr_decoder_lr_latent_cifar_optim.pt'
model_name = 'cifar'
if train_phase:
    net = VAE(model_name)
    net.init_model()
    for i in range(0, 210):
        net.fit_train(i)
        net.test(i)

        if i % 100 == 0 and i != 0:
            torch.save(
                net.state_dict(),
                "./checkpoints/vae_lr_decoder_lr_latent_" + model_name + f"_model_{str(i + 1).zfill(4)}.pt"
            )
            torch.save(
                net.optimizer.state_dict(),
                "./checkpoints/vae_lr_decoder_lr_latent_" + model_name + f"_optim_{str(i + 1).zfill(4)}.pt"
            )
            net.save_history(i + 1)
            net.save_to_zip(i + 1)



else:
    net = VAE("mnist")
    net.init_model()
    net.load_state_dict(torch.load(model_path))
    net.optimizer.load_state_dict(torch.load(optim_path))
    net.eval()
