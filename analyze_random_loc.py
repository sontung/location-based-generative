from os import listdir
from os.path import isfile, join
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from models import LocationBasedGenerator, LocationBasedGeneratorCoordConv
from data_loader import PBWrandom_loc, pbw_collate_fn
from utils import compute_iou
from torchvision.transforms import ToPILImage, ToTensor
import torch
import numpy as np
import sys
sys.path.append("ccl")
import ccl

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

def predict_for_different_masks(model_, start_, def_, wei_, true_graph_, ob_names_):
    with torch.no_grad():
        _, start_pred = model_(start_, def_, wei_)
        pred_sg = sae.return_sg(start_, [ob_names_])
    print(np.mean(compute_iou(start_pred, start_)[0]))
    print(sorted(true_graph_) == sorted(pred_sg[0]))


def roll_masks(masks2_, non_zeros):
    masks_ = masks2_.clone()
    possible_shifts = [0]
    rolled_masks = {sh: [] for sh in possible_shifts}

    for id_ in range(masks_.size(0)):
        first_ = tuple(masks_[id_].nonzero()[0][1:].tolist())
        if first_ in non_zeros:
            for sh in possible_shifts:
                rolled_masks[sh].append(torch.roll(masks_[id_], shifts=sh, dims=2))
    res_ = []
    for r_ in rolled_masks.values():
        res_.append(r_)
    return res_

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
to_pil = ToPILImage()
to_tensor = ToTensor()
for add in [0.5, 1, 2]:
    train_data = PBWrandom_loc(root_dir="/home/sontung/thesis/photorealistic-blocksworld/blocks-6-3", data_id=str(add),
                     nb_samples=100, train_size=1.0)
    train_iterator = DataLoader(train_data, batch_size=8, shuffle=True, collate_fn=pbw_collate_fn)
    train_batch = train_data[0]
    start1, coord_true, default, weight_maps = [tensor.to(device) for tensor in train_batch[:4]]
    graphs, ob_names = train_batch[4:]

    old_map = torch.sum(start1.squeeze(), dim=0)
    an_image = to_pil(old_map.cpu())
    bool_image = ccl.image_to_2d_bool_array(an_image)
    result, idx_set = ccl.connected_component_labelling(bool_image, 8)
    new_images = []
    for set_ in idx_set.values():
        im2 = np.zeros((128, 128, 3))
        new_images.append(roll_masks(train_data[0][0].squeeze(), set_))

    predict_for_different_masks(sae, start1, default, weight_maps, graphs, ob_names)
    for i1 in new_images[0]:
        for i2 in new_images[1]:
            for i3 in new_images[2]:
                t = torch.sum(torch.stack(i1+i2+i3, dim=0), dim=0)
                start2 = torch.stack(i1+i2+i3, dim=0).to(device).unsqueeze(0)
                predict_for_different_masks(sae, start2, default, weight_maps, graphs, ob_names)
    # to_pil(torch.sum(start1.cpu().squeeze(), dim=0)).show()
    # to_pil(torch.sum(start2.cpu().squeeze(), dim=0)).show()
    # print(torch.sum(torch.eq(start1, start2)))
    # print(torch.eq(start1, start2))
    # print(torch.sum(start1), torch.sum(start2))
    break
    # print("evaluating %d samples" % len(train_data))
    # print(eval_f(sae, train_iterator))
