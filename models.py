import torch.nn as nn

import torch
import torchvision
import torch.nn.functional as F
import torchvision.models as models

from PIL import Image


class RearrangeNetwork(nn.Module):
    def __init__(self):
        super(RearrangeNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(50460, 50)
        self.fc2 = nn.Linear(50, 10)

        self.action_emb = nn.Embedding(6, 32)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(7840+32, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 32),
            nn.ReLU(True),
            nn.Linear(32, 6)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def transform(self, x, theta):
        grid = F.affine_grid(theta.view(theta.size(0), 2, 3), x.size())
        x = F.grid_sample(x, grid)
        return x

    def find_theta(self, x0, act):
        im_features = self.localization(x0)
        im_features = im_features.view(im_features.size(0), -1)
        act_features = self.action_emb(torch.argmax(act, dim=1).to(x0.device))
        return self.fc_loc(torch.cat([im_features, act_features], dim=1))

    def infer(self, x0, act):
        nb_objects = x0.size(1)
        new_act = torch.zeros(act.size(0) * nb_objects, act.size(1))
        for i in range(act.size(0)):
            for j in range(nb_objects):
                new_act[i * nb_objects + j] = act[i]
        theta = self.find_theta(x0.view(-1, 3, 128, 128), new_act)
        x1_pred = self.transform(x0.view(-1, 3, 128, 128), theta).view(-1, nb_objects, 3, 128, 128)
        x1_pred_sum = torch.sum(x1_pred, dim=1)
        print(theta[:, (2, 5)])
        return x1_pred_sum

    def forward(self, x0, act, x1):
        pred = self.infer(x0, act)
        return nn.functional.mse_loss(pred, x1), pred


class LocationBasedGenerator(nn.Module):
    def __init__(self, output_dim=6, input_channels=6):
        super(LocationBasedGenerator, self).__init__()

        # resnet = models.resnet34(pretrained=True)
        # layers = list(resnet.children())
        #
        # # remove the last layer
        # layers.pop()
        # # remove the first layer as we take a 6-channel input
        # layers.pop(0)
        # layers.insert(0, nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False))
        #
        # self.main = nn.Sequential(*layers)
        # self.final_layer = nn.Sequential(
        #     nn.Linear(512, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, output_dim)
        # )
        # self.final_layer[-1].weight.data.zero_()
        # self.final_layer[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(6, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(7840, 32),
            nn.ReLU(True),
            nn.Linear(32, 6)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def transform(self, x, theta):
        x = x[:, :3, :, :]
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid)
        return x

    def find_theta(self, x):
        # x = self.main(x)
        # output = self.final_layer(x.squeeze())

        xs = self.localization(x)
        xs = xs.view(xs.size(0), -1)
        theta = self.fc_loc(xs)
        print(theta)

        return theta.view(-1, 2, 3)

    def infer(self, x, x_default):
        theta = self.find_theta(x)
        x_pred = self.transform(x_default, theta)
        return x_pred

    def forward(self, x, x_default):
        pred = self.infer(x, x_default)
        return nn.functional.mse_loss(pred, x[:, :3, :, :]), pred


class Transformer(nn.Module):
    def __init__(self, scalew=1, scaleh=1, transX=0., transY=0.):
        super().__init__()

        self.theta = nn.Parameter(torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ])[None], requires_grad=True)

    def forward(self, x):
        grid = F.affine_grid(self.theta, x.size(), align_corners=False)
        x_pred = F.grid_sample(x, grid, align_corners=False)
        return F.mse_loss(x_pred, x), x_pred


if __name__ == '__main__':
    model = RearrangeNetwork()