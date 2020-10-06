import sys
import time
import torch.nn as nn
import torch
import torchvision
import torch.nn.functional as F
from resnet import resnet34, CoordConv
from utils import recon_sg, find_nb_blocks
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



class LocationBasedGenerator(nn.Module):
    def __init__(self, output_dim=2, input_channels=3):
        super(LocationBasedGenerator, self).__init__()

        # Spatial transformer localization-network
        # resnet = resnet34(pretrained=False)
        resnet = torchvision.models.resnet34(pretrained=True)

        layers = list(resnet.children())

        # remove the last layer
        layers.pop()
        # remove the first layer as we take a 6-channel input
        layers.pop(0)
        layers.insert(0, nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False))

        self.main = nn.Sequential(*layers)
        self.final_layer = nn.Linear(512, output_dim)

        # Initialize the weights/bias with identity transformation
        self.final_layer.weight.data.zero_()
        self.final_layer.bias.data.copy_(torch.tensor([0.5, 0.5], dtype=torch.float))

    def transform(self, x, theta):
        x = x[:, :3, :, :]
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def find_theta(self, x):
        xs = self.main(x)
        if xs.size(0) == 1:
            xs = xs.squeeze().unsqueeze(0)
        else:
            xs = xs.squeeze()
        trans_vec = self.final_layer(xs)
        trans_vec = torch.sigmoid(trans_vec)
        theta = torch.tensor([1, 0, -1.6, 0, 1, 1.6], dtype=torch.float, requires_grad=True).repeat(x.size(0), 1).to(x.device)
        theta[:, 2] = -trans_vec[:, 0]*1.6
        theta[:, 5] = trans_vec[:, 1]*1.6
        return theta.view(-1, 2, 3)

    def infer(self, x, x_default):
        theta = self.find_theta(x)
        x_pred = self.transform(x_default, theta)
        return x_pred

    def return_sg(self, x, ob_names):
        """
        x: masks, (bs, nb masks, 3, 128, 128)
        ob_names: name for each mask
        """
        scene_img = torch.sum(x, dim=1)
        scene_img = torch.sum(scene_img, dim=1)
        nb_blocks_per_img = [find_nb_blocks(scene_img[du1]) for du1 in range(scene_img.size(0))]
        nb_objects = x.size(1)
        batch_size = x.size(0)
        x = x.view(-1, 3, 128, 128)
        with torch.no_grad():
            theta = self.find_theta(x).view(batch_size, nb_objects, 6)
        trans_vec = theta[:, :, [2, 5]]
        trans_vec[:, :, 0] *= -1
        res = []

        for i in range(batch_size):
            sg = recon_sg(ob_names[i], trans_vec[i], nb_blocks_per_img[i])
            res.append(sorted(sg))

        return res


    def forward(self, x, x_default, weights, using_weights=True, only_pred=False):
        nb_objects = x.size(1)

        # single
        x = x.view(-1, 3, 128, 128)
        x_default = x_default.view(-1, 3, 128, 128)
        weights = weights.view(-1, 128, 128)
        pred = self.infer(x, x_default)
        if only_pred:
            return pred
        pos_loss = nn.functional.mse_loss(pred, x[:, :3, :, :], reduction="none")

        if using_weights:
            pos_loss = torch.mean(pos_loss, dim=1) * weights.squeeze()
        else:
            pos_loss = torch.mean(pos_loss, dim=1)

        pos_loss = pos_loss.mean()
        zeros = torch.zeros_like(pos_loss)
        hinge = nn.functional.mse_loss(x_default, x[:, :3, :, :])
        neg_loss = torch.max(zeros,
                             torch.ones_like(pos_loss)*hinge - nn.functional.mse_loss(pred, x_default)).mean()

        # all
        x = x.view(-1, nb_objects, 3, 128, 128)
        pred = pred.view(-1, nb_objects, 3, 128, 128)
        scene_loss = nn.functional.mse_loss(torch.sum(pred, dim=1), torch.sum(x, dim=1))
        return pos_loss+neg_loss+scene_loss, pred

class LocationBasedGeneratorWithScaling(nn.Module):
    def __init__(self, output_dim=3, input_channels=3):
        super(LocationBasedGeneratorWithScaling, self).__init__()

        # Spatial transformer localization-network
        # resnet = resnet34(pretrained=False)
        resnet = torchvision.models.resnet34(pretrained=True)

        layers = list(resnet.children())

        # remove the last layer
        layers.pop()
        # remove the first layer as we take a 6-channel input
        layers.pop(0)
        layers.insert(0, nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False))

        self.main = nn.Sequential(*layers)
        self.final_layer = nn.Linear(512, output_dim)

        # Initialize the weights/bias with identity transformation
        # self.final_layer.weight.data.zero_()
        # self.final_layer.bias.data.copy_(torch.tensor([0.5, 0.5, 0.5], dtype=torch.float))

    def transform(self, x, theta):
        x = x[:, :3, :, :]
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def find_theta(self, x):
        xs = self.main(x)
        trans_vec = self.final_layer(xs.squeeze())
        trans_vec = torch.sigmoid(trans_vec)
        theta = torch.tensor([1, 0, -1.6, 0, 1, 1.6], dtype=torch.float, requires_grad=True).repeat(x.size(0), 1).to(x.device)
        theta[:, 2] = -trans_vec[:, 0]*1.6
        theta[:, 5] = trans_vec[:, 1]*1.6
        theta[:, 0] = trans_vec[:, 2]*1.9+0.1
        theta[:, 4] = trans_vec[:, 2]*1.9+0.1

        return theta.view(-1, 2, 3)

    def infer(self, x, x_default):
        theta = self.find_theta(x)
        print(theta[0])
        x_pred = self.transform(x_default, theta)
        return x_pred

    def return_sg(self, x, ob_names):
        """
        x: masks, (bs, nb masks, 3, 128, 128)
        ob_names: name for each mask
        """
        scene_img = torch.sum(x, dim=1)
        scene_img = torch.sum(scene_img, dim=1)
        nb_blocks_per_img = [find_nb_blocks(scene_img[du1]) for du1 in range(scene_img.size(0))]
        nb_objects = x.size(1)
        batch_size = x.size(0)
        x = x.view(-1, 3, 128, 128)
        with torch.no_grad():
            theta = self.find_theta(x).view(batch_size, nb_objects, 6)
        trans_vec = theta[:, :, [2, 5]]
        trans_vec[:, :, 0] *= -1
        res = []

        for i in range(batch_size):
            sg = recon_sg(ob_names[i], trans_vec[i], nb_blocks_per_img[i])
            res.append(sorted(sg))

        return res


    def forward(self, x, x_default, weights, using_weights=False, only_pred=False):
        nb_objects = x.size(1)

        # single
        x = x.view(-1, 3, 128, 128)
        x_default = x_default.view(-1, 3, 128, 128)
        weights = weights.view(-1, 128, 128)
        pred = self.infer(x, x_default)
        if only_pred:
            return pred
        pos_loss = nn.functional.mse_loss(pred, x[:, :3, :, :], reduction="none")

        if using_weights:
            pos_loss = torch.mean(pos_loss, dim=1) * weights.squeeze()
        else:
            pos_loss = torch.mean(pos_loss, dim=1)

        pos_loss = pos_loss.mean()
        zeros = torch.zeros_like(pos_loss)
        hinge = nn.functional.mse_loss(x_default, x[:, :3, :, :])
        neg_loss = torch.max(zeros,
                             torch.ones_like(pos_loss)*hinge - nn.functional.mse_loss(pred, x_default)).mean()

        # all
        x = x.view(-1, nb_objects, 3, 128, 128)
        pred = pred.view(-1, nb_objects, 3, 128, 128)
        scene_loss = nn.functional.mse_loss(torch.sum(pred, dim=1), torch.sum(x, dim=1))
        return pos_loss+neg_loss+scene_loss, pred


class LocationBasedGeneratorCoordConv(nn.Module):
    def __init__(self, output_dim=2, input_channels=3):
        super(LocationBasedGeneratorCoordConv, self).__init__()

        # Spatial transformer localization-network
        resnet = torchvision.models.resnet34(pretrained=True)
        layers = list(resnet.children())

        # remove the last layer
        layers.pop()
        # remove the first layer as we take a 6-channel input
        layers.pop(0)
        layers.insert(0, CoordConv(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False))

        self.main = nn.Sequential(*layers)
        self.final_layer = nn.Linear(512, output_dim)

        # Initialize the weights/bias with identity transformation
        self.final_layer.weight.data.zero_()
        self.final_layer.bias.data.copy_(torch.tensor([0.5, 0.5], dtype=torch.float))

    def transform(self, x, theta):
        x = x[:, :3, :, :]
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def find_theta(self, x):
        xs = self.main(x)
        trans_vec = self.final_layer(xs.squeeze())
        trans_vec = torch.sigmoid(trans_vec)
        theta = torch.tensor([1, 0, -1.6, 0, 1, 1.6], dtype=torch.float, requires_grad=True).repeat(x.size(0), 1).to(x.device)
        theta[:, 2] = -trans_vec[:, 0]*1.6
        theta[:, 5] = trans_vec[:, 1]*1.6
        return theta.view(-1, 2, 3)

    def infer(self, x, x_default):
        theta = self.find_theta(x)
        x_pred = self.transform(x_default, theta)
        return x_pred

    def return_sg(self, x, ob_names):
        """
        x: masks, (bs, nb masks, 3, 128, 128)
        ob_names: name for each mask
        """
        scene_img = torch.sum(x, dim=1)
        scene_img = torch.sum(scene_img, dim=1)
        nb_blocks_per_img = [find_nb_blocks(scene_img[du1]) for du1 in range(scene_img.size(0))]
        nb_objects = x.size(1)
        batch_size = x.size(0)
        x = x.view(-1, 3, 128, 128)
        with torch.no_grad():
            theta = self.find_theta(x).view(batch_size, nb_objects, 6)
        trans_vec = theta[:, :, [2, 5]]
        trans_vec[:, :, 0] *= -1
        res = []
        for i in range(batch_size):
            sg = recon_sg(ob_names[i], trans_vec[i], nb_blocks_per_img[i])
            res.append(sorted(sg))
        return res

    def forward(self, x, x_default, weights, using_weights=True):
        nb_objects = x.size(1)

        # single
        x = x.view(-1, 3, 128, 128)
        x_default = x_default.view(-1, 3, 128, 128)
        weights = weights.view(-1, 128, 128)
        pred = self.infer(x, x_default)
        pos_loss = nn.functional.mse_loss(pred, x[:, :3, :, :], reduction="none")

        if using_weights:
            pos_loss = torch.mean(pos_loss, dim=1) * weights.squeeze()
        else:
            pos_loss = torch.mean(pos_loss, dim=1)

        pos_loss = pos_loss.mean()
        zeros = torch.zeros_like(pos_loss)
        hinge = nn.functional.mse_loss(x_default, x[:, :3, :, :])
        neg_loss = torch.max(zeros,
                             torch.ones_like(pos_loss)*hinge - nn.functional.mse_loss(pred, x_default)).mean()

        # all
        x = x.view(-1, nb_objects, 3, 128, 128)
        pred = pred.view(-1, nb_objects, 3, 128, 128)
        scene_loss = nn.functional.mse_loss(torch.sum(pred, dim=1), torch.sum(x, dim=1))
        return pos_loss+neg_loss+scene_loss, pred



if __name__ == '__main__':
    model = LocationBasedGeneratorWithScaling()
    model.cuda()
    import skimage.draw
    import numpy as np
    from PIL import Image
    from torchvision.transforms import ToPILImage, ToTensor

    mat = np.zeros([128, 128, 3], dtype=np.uint8)
    cent=[64,64]
    size=7
    square = skimage.draw.polygon([cent[0] - size, cent[0] + size, cent[0] + size, cent[0] - size],
                                  [cent[1] - size, cent[1] - size, cent[1] + size, cent[1] + size])
    mat[square] = [66, 0, 192]
    pil = Image.fromarray(mat)
    pil.show()
    to_tensor = ToTensor()
    to_pil = ToPILImage()

    tensor = to_tensor(pil).unsqueeze(0).cuda()
    for s in range(20):
        s/=10
        print(s)
        theta = torch.tensor([s, 0, 0, 0, s, 0], dtype=torch.float, requires_grad=True).repeat(tensor.size(0), 1).to(
            tensor.device)
        theta = theta.view(-1, 2, 3)
        im = to_pil(model.transform(tensor, theta).squeeze().cpu())
        plt.imshow(im)
        plt.show()
        plt.clf()  # will make the plot window empty
        im.close()
