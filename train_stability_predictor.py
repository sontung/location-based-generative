import numpy as np
import skimage.draw
import torch
import torch.nn as nn
import torchvision
from itertools import product
from PIL import Image
from data_loader import PBWstability_data
from torch.utils.data import DataLoader


def draw_circle(mat, cent, color):
    circle = skimage.draw.circle(cent[0], cent[1], 10)
    mat[circle] = color
    return mat, min(circle[0])

def draw_sq(mat, cent, color, size=10):
    square = skimage.draw.polygon([cent[0]-size, cent[0]+size, cent[0]+size, cent[0]-size],
                                  [cent[1]-size, cent[1]-size, cent[1]+size, cent[1]+size])
    mat[square] = color

    return mat, min(square[0])

def draw_tri(mat, cent, color, size=10):
    square = skimage.draw.polygon([cent[0]- size, cent[0] + size, cent[0] + size],
                                  [cent[1], cent[1] - size, cent[1] + size],
                                  )
    mat[square] = color
    return mat, min(square[0])

shape2im = {
    "circle": (draw_circle, [66, 128, 64]),
    "sq": (draw_sq, [194, 0, 64]),
    "tri": (draw_tri, [187, 187, 253])
}

class StabilityPredictor(nn.Module):
    def __init__(self):
        super(StabilityPredictor, self).__init__()
        # self.pretrained_model = torchvision.models.resnet34(pretrained=True)

        self.final_layer = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(1000, 250),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(250, 50),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(50, 8),
        )

    def forward(self, image_, pretrained_model_):
        return self.final_layer(pretrained_model_(image_))

def sg2im(stack):
    pos = 216
    image = np.zeros([224, 224, 3], dtype=np.uint8)
    for obj in stack:
        _, pos = shape2im[obj][0](image, [pos-11, 224/2], shape2im[obj][1])
    return image

def find_target(seq_):
    target_ = 0
    for idx_, i in enumerate(list(reversed(seq_))):
        if i in ["tri", "circle"]:
            target_ = idx_
    return target_

def generate_data(max_nb_objects=7):
    objects = ["circle", "tri", "sq"]
    im_id = 0
    for nb_objects in range(2, max_nb_objects+1):
        all_seq = list(product(objects, repeat=nb_objects))
        for seq in all_seq:
            im_id += 1
            target = find_target(seq)
            im_arr = sg2im(seq)
            im_name = "data/stability/im%d-stable%d.png" % (im_id, target)
            Image.fromarray(im_arr).save(im_name)

def collate_fn(batch):
    im_all, tar_all = [], []
    for all_inp_, tar_ in batch:
        im_all.append(all_inp_.unsqueeze(0))
        tar_all.append(tar_)

    im_all = torch.cat(im_all)
    tar_all = torch.tensor(tar_all)
    assert im_all.size(0) == tar_all.size(0)
    return im_all, tar_all

# generate_data()
train_data = PBWstability_data(train=True, train_size=0.7)
train_iterator = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_data = PBWstability_data(train=False, train_size=0.7)
test_iterator = DataLoader(test_data, batch_size=32, shuffle=True, collate_fn=collate_fn)

device = "cuda:0"

net = StabilityPredictor()
net.to(device)
pretrained_model = torchvision.models.resnet34(pretrained=True)
pretrained_model.eval()
pretrained_model.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

for epoch_id in range(20):
    net.train()
    corrects = 0.0
    nb_samples = 0
    for batch_id, batch in enumerate(train_iterator):
        optimizer.zero_grad()

        images, targets = [tensor.to(device) for tensor in batch]
        preds = net(images, pretrained_model)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()

        nb_samples += targets.size(0)
        corrects += torch.sum(torch.argmax(preds, dim=1)==targets).item()
    print("train acc:", corrects/nb_samples, corrects, nb_samples)

    net.eval()
    corrects = 0.0
    nb_samples = 0
    for batch_id, batch in enumerate(test_iterator):
        images, targets = [tensor.to(device) for tensor in batch]
        with torch.no_grad():
            preds = net(images, pretrained_model)
            loss = criterion(preds, targets)
        corrects += torch.sum(torch.argmax(preds, dim=1)==targets).item()
        nb_samples += targets.size(0)
    print("test acc:", corrects/nb_samples, corrects, nb_samples)
