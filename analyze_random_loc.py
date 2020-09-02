from os import listdir
from os.path import isfile, join
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from models import LocationBasedGenerator, LocationBasedGeneratorCoordConv
from data_loader import PBWrandom_loc, pbw_collate_fn
from utils import compute_iou, show2, return_default_mat
from torchvision.transforms import ToPILImage, ToTensor
import torch
import numpy as np
import sys
sys.path.append("ccl")
import ccl
import time

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

def predict_for_different_masks(model_, start_, start_prev, def_, wei_, true_graph_, ob_names_, z=""):
    if z == "second":
        def_ = []
        for idx_ in range(start_.size(1)):
            def_.append(return_default_mat(start_[0][idx_].cpu())[0].unsqueeze(0))
        def_ = torch.cat(def_, dim=0).unsqueeze(0).to(start_.device)
    with torch.no_grad():
        start_pred = model_(start_, def_, wei_, only_pred=True)
        pred_sg = sae.return_sg(start_, [ob_names_])
    # print(np.mean(compute_iou(start_pred, start_)[0]), sorted(true_graph_) == sorted(pred_sg[0]))
    return np.mean(compute_iou(start_pred, start_)[0]), pred_sg[0]

def find_cc(start_):
    old_map = torch.sum(start_.squeeze(), dim=0)
    an_image = to_pil(old_map.to("cpu"))
    bool_image = ccl.image_to_2d_bool_array(an_image)
    result, idx_set = ccl.connected_component_labelling(bool_image, 8)
    return result, idx_set

def roll_masks(masks2_, non_zeros, names):
    masks_ = masks2_.clone()
    possible_shifts = [0, -5, 5, -10, 10, -15, 15]
    rolled_masks = {sh: [] for sh in possible_shifts}
    for id_ in range(masks_.size(0)):
        first_ = tuple(masks_[id_].nonzero()[0][1:].tolist())
        if first_ in non_zeros:
            for sh in possible_shifts:
                rolled_masks[sh].append([torch.roll(masks_[id_], shifts=sh, dims=2), names[id_]])
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

device = "cuda:1"
sae = LocationBasedGenerator()
sae.to(device)
sae.load_state_dict(torch.load("pre_models/model-20200726-041935", map_location=device))
sae.eval()
to_pil = ToPILImage()
to_tensor = ToTensor()
for add in [0.5, 1, 2]:
    train_data = PBWrandom_loc(root_dir="/home/sontung/thesis/photorealistic-blocksworld/blocks-6-3", data_id=str(add),
                     nb_samples=500, train_size=1.0)
    train_iterator = DataLoader(train_data, batch_size=8, shuffle=True, collate_fn=pbw_collate_fn)

    all_ious1 = []
    all_ious2 = []
    corr1 = 0
    corr2 = 0

    for index in range(500):
        train_batch = train_data[index]

        start1 = train_batch[0]
        default = train_batch[2]
        weight_maps = train_batch[3]
        graphs, ob_names = train_batch[4:]

        start1 = start1.to(device)
        default = default.to(device)

        result, idx_set = find_cc(start1)
        number_of_stacks_original = len(idx_set)
        new_images = []
        for set_ in idx_set.values():
            new_images.append(roll_masks(start1.squeeze(), set_, ob_names))

        ori_iou, ori_pred_sg = predict_for_different_masks(sae, start1, start1, default, weight_maps, graphs, ob_names)
        best_sec_iou = 0
        best_sg = None

        for i1 in new_images[0]:
            for i2 in new_images[1]:
                for i3 in new_images[2]:
                    tensors = [du33[0] for du33 in i1+i2+i3]
                    ob_names = [du33[1] for du33 in i1+i2+i3]
                    start2 = torch.stack(tensors, dim=0).unsqueeze(0)
                    number_of_stacks_second = len(find_cc(start2)[1])
                    if number_of_stacks_original != number_of_stacks_second:  # break structure
                        continue
                    sec_iou, sec_sg = predict_for_different_masks(sae, start2, start1, default,
                                                                   weight_maps, graphs, ob_names, z="second")
                    if best_sg is None:
                        best_sec_iou = sec_iou
                        best_sg = sec_sg
                    else:
                        if sec_iou > best_sec_iou:
                            best_sec_iou = sec_iou
                            best_sg = sec_sg

        all_ious1.append(max([ori_iou, best_sec_iou]))
        all_ious2.append(ori_iou)
        if sorted(graphs) == sorted(best_sg):
            corr1 += 1
        if sorted(graphs) == sorted(ori_pred_sg):
            corr2 += 1
    print("original iou:", np.mean(all_ious2), corr2/500)
    print("improved iou:", np.mean(all_ious1), corr1/500)


    # to_pil(torch.sum(start1.cpu().squeeze(), dim=0)).show()
    # to_pil(torch.sum(start2.cpu().squeeze(), dim=0)).show()
    # print(torch.sum(torch.eq(start1, start2)))
    # print(torch.eq(start1, start2))
    # print(torch.sum(start1), torch.sum(start2))
    # break
    # print("evaluating %d samples" % len(train_data))
    # print(eval_f(sae, train_iterator))
