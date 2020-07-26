import json
import torch
import torchvision
import numpy as np
import pickle
import time
import random
import math
import sys
from utils import read_seg_masks
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from os import listdir
from os.path import isfile, join
from PIL import Image


class PBW(Dataset):
    def __init__(self, root_dir="/home/sontung/thesis/photorealistic-blocksworld/blocks-6-3",
                 train=True, train_size=0.6, nb_samples=1000, json2im=None, if_save_data=True):
        print("Loading from", root_dir)
        super(PBW, self).__init__()
        self.root_dir = root_dir
        self.train = train
        identifier = root_dir.split("/")[-1]

        json_dir = "%s/scene" % root_dir
        self.scene_jsons = [join(json_dir, f) for f in listdir(json_dir) if isfile(join(json_dir, f))]

        self.image_dir = "%s/image" % root_dir
        self.transform = torchvision.transforms.ToTensor()
        self.transform_to_pil = torchvision.transforms.ToPILImage()

        if isfile("data/json2sg-%s" % identifier):
            print("Loading precomputed json2sg:", "data/json2sg-%s" % identifier)
            with open("data/json2sg-%s" % identifier, 'rb') as f:
                self.json2sg = pickle.load(f)
        else:
            self.json2sg = {}
            for js in self.scene_jsons:
                self.json2sg[js] = read_scene_json(js)
            if if_save_data:
                with open("data/json2sg-%s" % identifier, 'wb') as f:
                    pickle.dump(self.json2sg, f, pickle.HIGHEST_PROTOCOL)

        if json2im is None:
            self.json2im = self.load_json2im(nb_samples=nb_samples)
        else:
            self.json2im = json2im

        keys = list(self.json2im.keys())
        if train:
            self.data = {du3: self.json2im[du3] for du3 in keys[:int(len(keys)*train_size)]}
        else:
            self.data = {du3: self.json2im[du3] for du3 in keys[int(len(keys)*train_size):]}
        self.keys = list(self.data.keys())
        print("loaded", len(self.json2im))

    def load_json2im(self, nb_samples=1000):
        if nb_samples < 0:
            nb_samples = len(self.scene_jsons)
        name = "%s-%d" % (self.root_dir.split("/")[-1], nb_samples)
        if isfile("data/%s" % name):
            print("Loading precomputed json2im:", "data/%s" % name)
            with open("data/%s" % name, 'rb') as f:
                return pickle.load(f)
        else:
            res_dict = {}
            if nb_samples > 0:
                random.shuffle(self.scene_jsons)
            for item in range(len(self.scene_jsons))[:nb_samples]:
                bboxes, coords, obj_names, img_name = self.json2sg[self.scene_jsons[item]]
                sg = recon_sg2(self.scene_jsons[item])

                img_pil = Image.open("%s/%s" % (self.image_dir, img_name)).convert('RGB')
                img = self.transform(img_pil).unsqueeze(0)
                all_inp = []
                original_inp = []
                weight_maps = []

                for bbox in bboxes:
                    mask_im = torch.zeros_like(img)
                    default_inp = torch.zeros_like(img)

                    bbox_int = [int(dm11) for dm11 in bbox]

                    # masked the image
                    for x_ in range(bbox_int[0], bbox_int[2]+1):
                        idc = range(bbox_int[1], bbox_int[3]+1)
                        mask_im[:, :, idc, x_] = img[:, :, idc, x_]

                    # place the mask at the origin
                    idc1 = range(bbox_int[0], bbox_int[2]+1)
                    idc2 = range(len(idc1))
                    for j_, y_ in enumerate(range(bbox_int[1], bbox_int[3]+1)):
                        default_inp[:, :, j_+128-int(-bbox[1]+bbox[3]+1), idc2] = img[:, :, y_, idc1]

                    weight = construct_weight_map(bbox)

                    all_inp.append(mask_im)
                    original_inp.append(default_inp)
                    weight_maps.append(weight)

                targets = torch.from_numpy(np.array(coords)).float().flatten()
                all_inp = torch.cat(all_inp, dim=0).unsqueeze(0)
                original_inp = torch.cat(original_inp, dim=0).unsqueeze(0)
                weight_maps = torch.cat(weight_maps, dim=0).unsqueeze(0)

                res_dict[self.scene_jsons[item]] = (all_inp, targets, original_inp, weight_maps, sg, obj_names)
            with open("data/%s" % name, 'wb') as f:
                pickle.dump(res_dict, f, pickle.HIGHEST_PROTOCOL)
            return res_dict

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, item):
        return self.data[self.keys[item]]


class PBW_Planning_only(Dataset):
    def __init__(self, root_dir="/home/sontung/thesis/photorealistic-blocksworld/blocks-6-3",
                 train=True, train_size=0.6, nb_samples=1000, json2im=None):
        print("Loading from", root_dir)
        super(PBW_Planning_only, self).__init__()
        self.root_dir = root_dir
        self.train = train
        identifier = root_dir.split("/")[-1]

        json_dir = "%s/scene" % root_dir
        self.scene_jsons = [join(json_dir, f) for f in listdir(json_dir) if isfile(join(json_dir, f))]

        self.image_dir = "%s/image" % root_dir
        self.transform = torchvision.transforms.ToTensor()
        self.transform_to_pil = torchvision.transforms.ToPILImage()

        if isfile("data/json2sg-%s" % identifier):
            print("Loading precomputed json2sg:", "data/json2sg-%s" % identifier)
            with open("data/json2sg-%s" % identifier, 'rb') as f:
                self.json2sg = pickle.load(f)
        else:
            self.json2sg = {}
            for js in self.scene_jsons:
                self.json2sg[js] = read_scene_json(js)
            with open("data/json2sg-%s" % identifier, 'wb') as f:
                pickle.dump(self.json2sg, f, pickle.HIGHEST_PROTOCOL)

        if json2im is None:
            self.json2im = self.load_json2im(nb_samples=nb_samples)
        else:
            self.json2im = json2im

        keys = list(self.json2im.keys())
        if train:
            self.data = {du3: self.json2im[du3] for du3 in keys[:int(len(keys)*train_size)]}
        else:
            self.data = {du3: self.json2im[du3] for du3 in keys[int(len(keys)*train_size):]}
        self.keys = list(self.data.keys())
        print("loaded", len(self.json2im))

    def load_json2im(self, nb_samples=1000):
        if nb_samples < 0:
            nb_samples = len(self.scene_jsons)
        name = "%s-%d-planning" % (self.root_dir.split("/")[-1], nb_samples)
        if isfile("data/%s" % name):
            print("Loading precomputed json2im:", "data/%s" % name)
            with open("data/%s" % name, 'rb') as f:
                return pickle.load(f)
        else:
            res_dict = {}
            for item in range(len(self.scene_jsons))[:nb_samples]:
                bboxes, coords, obj_names, img_name = self.json2sg[self.scene_jsons[item]]
                sg = recon_sg2(self.scene_jsons[item])

                img_pil = Image.open("%s/%s" % (self.image_dir, img_name)).convert('RGB')
                img = self.transform(img_pil).unsqueeze(0)
                all_inp = []

                for bbox in bboxes:
                    mask_im = torch.zeros_like(img)
                    bbox_int = [int(dm11) for dm11 in bbox]

                    # masked the image
                    for x_ in range(bbox_int[0], bbox_int[2]+1):
                        idc = range(bbox_int[1], bbox_int[3]+1)
                        mask_im[:, :, idc, x_] = img[:, :, idc, x_]

                    all_inp.append(mask_im)

                all_inp = torch.cat(all_inp, dim=0).unsqueeze(0)  # masks

                res_dict[self.scene_jsons[item]] = (all_inp, sg, obj_names)
            with open("data/%s" % name, 'wb') as f:
                pickle.dump(res_dict, f, pickle.HIGHEST_PROTOCOL)
            return res_dict

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, item):
        return self.data[self.keys[item]]

class SimData(Dataset):
    def __init__(self, root_dir="/home/sontung/Downloads/5objs_seg", nb_samples=10,
                 train=True, train_size=0.6, if_save_data=True):
        print("Loading from", root_dir)
        super(SimData, self).__init__()
        self.root_dir = root_dir
        self.train = train
        identifier = root_dir.split("/")[-1]

        self.scene_jsons = [join(root_dir, f) for f in listdir(root_dir) if isfile(join(root_dir, f))]
        print("there are %d files total" % len(self.scene_jsons))
        if nb_samples > 0:
            random.shuffle(self.scene_jsons)
            self.scene_jsons = self.scene_jsons[:nb_samples]

        name = "json2sg-%s-%d" % (identifier, nb_samples)
        if isfile("data/%s" % name):
            print("Loading precomputed json2sg:", "data/%s" % name)
            with open("data/%s" % name, 'rb') as f:
                self.js2data = pickle.load(f)
        else:
            self.js2data = {}
            for js in self.scene_jsons:
                self.js2data[js] = read_seg_masks(js)
            if if_save_data:
                with open("data/%s" % name, 'wb') as f:
                    pickle.dump(self.js2data, f, pickle.HIGHEST_PROTOCOL)

        keys = list(self.js2data.keys())
        if train:
            self.data = {du3: self.js2data[du3] for du3 in keys[:int(len(keys)*train_size)]}
        else:
            self.data = {du3: self.js2data[du3] for du3 in keys[int(len(keys)*train_size):]}
        self.keys = list(self.data.keys())
        print("loaded", len(self.js2data), "used", len(self.data))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, item):
        return self.data[self.keys[item]]

def construct_weight_map(bbox):
    weight_map = torch.ones(128, 128)
    a_ = 255
    b_ = 1
    bbox_int = [int(dm11) for dm11 in bbox]

    weight_map[:, range(0, bbox_int[0]+1)] *= torch.tensor([(b_-a_)*x_/bbox[0]+a_ for x_ in range(0, bbox_int[0]+1)])
    weight_map[:, range(bbox_int[2], 128)] *= torch.tensor([(a_-b_)*(x_-bbox[2])/(128-bbox[2])+b_ for x_ in range(bbox_int[2], 128)])

    for x_ in range(0, bbox_int[1]+1):
        weight_map[x_, :] *= (b_-a_)*x_/bbox[1]+a_
    for x_ in range(bbox_int[3], 128):
        weight_map[x_, :] *= (a_-b_)*(x_-bbox[3])/(128-bbox[3])+b_

    for x_ in range(bbox_int[0], bbox_int[2] + 1):
        for y_ in range(bbox_int[1], bbox_int[3] + 1):
            weight_map[y_, x_] = 0

    weight_map = torch.sqrt(weight_map)
    return weight_map.unsqueeze(0)

def read_scene_json2(json_file_dir):
    with open(json_file_dir, 'r') as json_file:
        du = json.load(json_file)
    return du["image_filename"]

def recon_sg2(json_file_dir):
    """
    reconstruct a sg from a scene json file
    """
    id2color = {
        "gray": [87, 87, 87],
        "red": [173, 35, 35],
        "blue": [42, 75, 215],
        "green": [29, 105, 20],
        "brown": [129, 74, 25],
        "purple": [129, 38, 192],
        "cyan": [41, 208, 208],
        "yellow": [255, 238, 51],
        "c1": [42, 87, 9],
        "c2": [255, 102, 255]
    }
    color2id = {tuple(v): u for u, v in id2color.items()}
    with open(json_file_dir, 'r') as json_file:
        du = json.load(json_file)
    location_dict = {}
    objects = []
    bboxes = []
    for obj in du["objects"]:
        color = tuple([int(du33*255) for du33 in obj["color"]][:-1])
        object_id = color2id[color]
        a_key = "%.3f" % obj["location"][0]
        if a_key not in location_dict:
            location_dict[a_key] = [(object_id, obj["location"][2])]
        else:
            location_dict[a_key].append((object_id, obj["location"][2]))
        objects.append(object_id)
        bboxes.append([
            obj["bbox"][0]/128.0,
            obj["bbox"][1]/128.0,
            obj["bbox"][2]/128.0,
            obj["bbox"][3]/128.0,
            ])
    obj2id = {objects[du4]: objects[du4] for du4 in range(len(objects))}
    relationships = [
        [obj2id["brown"], "left", obj2id["purple"]],
        [obj2id["purple"], "left", obj2id["cyan"]],
    ]
    for du3 in location_dict:
        location = sorted(location_dict[du3], key=lambda x: x[1])
        while len(location) > 1:
            o1 = location.pop()[0]
            o2 = location[-1][0]
            relationships.append([obj2id[o1], "up", obj2id[o2]])
            assert o1 not in ["cyan", "purple", "brown"]

    return relationships

def recon_sg(obj_names, locations, if_return_assigns=False):
    """
    reconstruct a sg from object names and coordinates
    """
    location_dict = {}
    objects = []

    if type(locations) == torch.Tensor:
        locations = locations.cpu().numpy()
    elif isinstance(locations, list):
        locations = np.array(locations)

    locations = locations.reshape(-1, 2)
    k_means_assign = kmeans(locations[:, 0])

    for idx, object_id in enumerate(obj_names):
        a_key = k_means_assign[idx]
        if a_key not in location_dict:
            location_dict[a_key] = [(object_id, locations[idx][1])]
        else:
            location_dict[a_key].append((object_id, locations[idx][1]))
        objects.append(object_id)
    relationships = [
        ["brown", "left", "purple"],
        ["purple", "left", "cyan"],
    ]
    for du3 in location_dict:
        location = sorted(location_dict[du3], key=lambda x: x[1])
        while len(location) > 1:
            o1 = location.pop()[0]
            o2 = location[-1][0]
            relationships.append([o1, "up", o2])
    if if_return_assigns:
        return relationships, k_means_assign
    return relationships

def check(d1, d2):
    for i in range(len(d1)):
        if d1[i] != d2[i]:
            print(d1)
            print(d2)
            print(d1[i], d2[i], i)
            return False
    return True

def sim_collate_fn(batch):
    all_imgs, all_imgs2, weights = [], [], []
    sgs = []
    names = []
    im_names = []
    for masks, def_mat, def_wei, ob_names, sg, im_name in batch:
        all_imgs.append(masks.unsqueeze(0))
        all_imgs2.append(def_mat.unsqueeze(0))
        weights.append(def_wei.unsqueeze(0))
        sgs.append(sg)
        names.append(ob_names)
        im_names.append(im_name)

    all_imgs = torch.cat(all_imgs)
    all_imgs2 = torch.cat(all_imgs2)
    weights = torch.cat(weights)

    return all_imgs, all_imgs2, weights, sgs, names, im_names

def pbw_collate_fn(batch):
    all_imgs, all_targets, all_imgs2, weights = [], [], [], []
    sgs = []
    names = []
    for i, (inp, tar, inp2, w, sg, name) in enumerate(batch):
        all_imgs.append(inp)
        all_imgs2.append(inp2)
        all_targets.append(tar.unsqueeze(0))
        weights.append(w)
        sgs.append(sg)
        names.append(name)

    all_imgs = torch.cat(all_imgs)
    all_imgs2 = torch.cat(all_imgs2)
    all_targets = torch.cat(all_targets)
    weights = torch.cat(weights)

    return all_imgs, all_targets, all_imgs2, weights, sgs, names

def collate_fn_trans(batch):
    all_imgs1, all_imgs2, all_act = [], [], []
    for start, act_onehot, end in batch:
        all_imgs1.append(start.unsqueeze(0))
        all_imgs2.append(end.unsqueeze(0))
        all_act.append(act_onehot.unsqueeze(0))

    all_imgs1 = torch.cat(all_imgs1)
    all_imgs2 = torch.cat(all_imgs2)
    all_act = torch.cat(all_act).float()

    return all_imgs1, all_imgs2, all_act

def evaluation(json2im, model, loss_func, device="cuda"):
    val_loss = []
    correct = 0
    total = 0.0
    best_p = None
    best_l = None
    best_a = None
    print("evaluating %d samples" % len(json2im))

    all_images = []
    for val_json in json2im:
        val_batch = json2im[val_json][:2]
        images, coords = [tensor.to(device).double() for tensor in val_batch]
        all_images.append(images)
    all_images = torch.cat(all_images, dim=0)
    im_iter = DataLoader(all_images, batch_size=64)
    all_pred_coords = []
    for im in im_iter:
        with torch.no_grad():
            pred_coords = model(im)
            all_pred_coords.append(pred_coords)
    all_pred_coords = torch.cat(all_pred_coords, dim=0)

    for idx, val_json in enumerate(json2im):
        val_batch = json2im[val_json][:2]
        obj_names = json2im[val_json][2]
        images, coords = [tensor.to(device).double() for tensor in val_batch]
        sg = recon_sg(obj_names, coords)

        pred_coords = all_pred_coords[idx]
        pred_sg, assigns = recon_sg(obj_names, pred_coords, if_return_assigns=True)
        correct += sg == pred_sg
        total += 1
        loss = loss_func(pred_coords, coords)
        val_loss.append(loss.item())
        if best_l is None or loss.item() < best_l:
            best_l = loss.item()
            best_p = (pred_coords, coords)
            best_a = assigns
    print("acquire this best:")
    print("assigns", best_a)
    print("pred\n", best_p[0])
    print("true\n", best_p[1])
    print()
    acc = correct / total
    return np.mean(val_loss), acc

def evaluation2(iter_, model, loss_func, device="cuda"):
    val_loss = []
    for val_batch in iter_:
        images, coords = [tensor.to(device).double() for tensor in val_batch]

        with torch.no_grad():
            pred_coords = model(images)

        loss = loss_func(pred_coords, coords)
        val_loss.append(loss.item())
    return np.mean(val_loss)

def find_top(up_rel, ob_start):
    rel_res = None
    while True:
        done = True
        for rel in up_rel:
            if rel[2] == ob_start:
                ob_start = rel[0]
                done = False
                rel_res = rel
                break
        if done:
            return ob_start, rel_res

def read_scene_json(json_file_dir, return_top_objects=False):
    from random import shuffle

    id2color = {
        "gray": [87, 87, 87],
        "red": [173, 35, 35],
        "blue": [42, 75, 215],
        "green": [29, 105, 20],
        "brown": [129, 74, 25],
        "purple": [129, 38, 192],
        "cyan": [41, 208, 208],
        "yellow": [255, 238, 51],
        "c1": [42, 87, 9],
        "c2": [255, 102, 255]
    }
    color2id = {tuple(v): u for u, v in id2color.items()}
    with open(json_file_dir, 'r') as json_file:
        du = json.load(json_file)
    location_dict = {}
    objects = []
    bboxes = []
    locations = []

    for obj in du["objects"]:
        color = tuple([int(du33*255) for du33 in obj["color"]][:-1])
        object_id = color2id[color]
        a_key = "%.3f" % obj["location"][0]
        if a_key not in location_dict:
            location_dict[a_key] = [(object_id, obj["location"][2])]
        else:
            location_dict[a_key].append((object_id, obj["location"][2]))
        objects.append(object_id)
        bboxes.append([
            obj["bbox"][0],
            obj["bbox"][1],
            obj["bbox"][2],
            obj["bbox"][3],
        ])
        locations.append([obj["location"][0], obj["location"][2]])
    sg = recon_sg(objects, locations)

    up_rel = [rel for rel in sg if rel[1] == "up"]
    values_for_masks = [dm1+1 for dm1 in list(range(len(objects)))]
    shuffle(values_for_masks)
    obID2mask_value = {objects[dm3]: values_for_masks[dm3] for dm3 in range(len(objects))}
    ob_mask_value = [obID2mask_value[dm4] for dm4 in objects]
    top_obj_ids = [obID2mask_value[find_top(up_rel, du)[0]]-1 for du in ["brown", "purple", "cyan"]]
    if return_top_objects:
        return bboxes, locations, objects, du["image_filename"], ob_mask_value, top_obj_ids
    return bboxes, locations, objects, du["image_filename"]

def kmeans(data_):
    c1 = max(data_)
    c2 = min(data_)
    c3 = (c1+c2)/2
    c_list = [c1, c2, c3]
    init_c_list = c_list[:]
    assign = [0]*len(data_)
    # print(data_)
    # print(data_.shape)
    for _ in range(10):
        for idx, d in enumerate(data_):
            assign[idx] = min([0, 1, 2], key=lambda x: (c_list[x]-d)**2)

        for c in range(3):
            stuff = [d for idx, d in enumerate(data_) if assign[idx] == c]
            if len(stuff) > 0:
                c_list[c] = sum(stuff)/len(stuff)
    # print(sorted(c_list), sorted(init_c_list), data_)
    return assign


if __name__ == '__main__':
    d = SimData()
