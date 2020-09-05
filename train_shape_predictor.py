import torch
import torch.nn as nn
import torchvision.models
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision.transforms import ToPILImage, Compose, Normalize, Resize, ToTensor
from data_loader import PBWshape_data

def collate_fn(batch):
    shape2id = {
        "SmoothCube_v2": 0,
        "Sphere": 1,
        "SmoothCylinder": 2
    }
    trans_ = Compose([
        ToPILImage(),
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    im_all, shape_all = [], []
    for all_inp_, shapes_, names_, sg_ in batch:
        all_inp_ = all_inp_.squeeze()
        for idx_ in range(all_inp_.size(0)):
            im_all.append(trans_(all_inp_[idx_].squeeze()).unsqueeze(0))
        shape_all.extend(map(lambda l_: shape2id[l_], shapes_))

    im_all = torch.cat(im_all)
    shape_all = torch.tensor(shape_all)
    assert im_all.size(0) == shape_all.size(0)
    return im_all, shape_all

class ShapePredictor(nn.Module):
    def __init__(self):
        super(ShapePredictor, self).__init__()
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
            nn.Linear(50, 3),
        )

    def forward(self, image_, pretrained_model_):
        return self.final_layer(pretrained_model_(image_))

device = "cuda:0"
train_data = PBWshape_data(nb_samples=-1, train=True, train_size=1.0)
train_iterator = DataLoader(train_data, batch_size=8, shuffle=True, collate_fn=collate_fn)
test_data = PBWshape_data(root_dir="/home/sontung/thesis/photorealistic-blocksworld/blocks-4-3-shape2",
                          nb_samples=-1, train=True, train_size=1.0)
test_iterator = DataLoader(train_data, batch_size=8, shuffle=True, collate_fn=collate_fn)

net = ShapePredictor()
net.to(device)
pretrained_model = torchvision.models.resnet34(pretrained=True)
pretrained_model.eval()
pretrained_model.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

for epoch_id in range(10):
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
