from os import listdir
from os.path import isfile, join
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from models import LocationBasedGenerator
from data_loader import PBWrandom_loc, pbw_collate_fn
from utils import compute_grad, show2, compute_iou

import torchvision
import torch
import numpy as np

def eval_f(model_, iter_, device_="cuda"):
    total_loss = 0
    correct = 0
    ious = []
    for idx, train_batch in enumerate(iter_):
        start, coord_true, default, weight_maps = [tensor.to(device_) for tensor in train_batch[:4]]
        graphs, ob_names = train_batch[4:]
        with torch.no_grad():
            loss, start_pred = model_(start, default, weight_maps)
            pred_sg = model_.return_sg(start, ob_names)
        ious.extend(compute_iou(start_pred, start)[0])
        total_loss += loss.item()
        for i in range(len(graphs)):
            correct += sorted(graphs[i]) == sorted(pred_sg[i])
    return total_loss, correct, np.mean(ious)

mypath = "/home/sontung/thesis/photorealistic-blocksworld/blocks-6-3/image"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

data_dict = {}
for f in onlyfiles:
    k = f.split("_")[-1]
    if k not in data_dict:
        data_dict[k] = [f]
    else:
        data_dict[k].append(f)

device = "cuda"
sae = LocationBasedGenerator()
sae.to(device)
sae.load_state_dict(torch.load("pre_models/model-20200726-041935", map_location=device))
sae.eval()

for add in [0.5, 1, 2]:
    train_data = PBWrandom_loc(root_dir="/home/sontung/thesis/photorealistic-blocksworld/blocks-6-3", data_id=str(add),
                     nb_samples=100, train_size=1.0)
    train_iterator = DataLoader(train_data, batch_size=8, shuffle=True, collate_fn=pbw_collate_fn)
    print("evaluating %d samples" % len(train_data))
    print(eval_f(sae, train_iterator))