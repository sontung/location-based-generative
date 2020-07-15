import json
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import Dataset, DataLoader

from os import listdir
from os.path import isfile, join

class PBWtransition(Dataset):
    def __init__(self, transition_dir="/home/sontung/thesis/photorealistic-blocksworld/blocks-6-3/image_tr",
                 train=True, nb_samples=-1, name2im=None):
        print("loading data from", transition_dir)
        onlyfiles = sorted([f for f in listdir(transition_dir) if isfile(join(transition_dir, f))])
        self.img_dir = transition_dir.replace("image_tr", "image")
        transition_js_dir = transition_dir.replace("image_tr", "scene_tr")

        transitions = {}
        self.action_list = {'01': 0, '02': 1, '10': 2, '12': 3, '20': 4, '21': 5}
        for i in range(0, len(onlyfiles), 2):
            first = onlyfiles[i]
            second = onlyfiles[i+1]
            first_js = "_".join(first.split("_")[:-1])+".json"
            second_js = "_".join(second.split("_")[:-1])+".json"

            _, loc1, ob1, im1 = read_scene_json(join(transition_js_dir, first_js))
            _, loc2, ob2, im2 = read_scene_json(join(transition_js_dir, second_js))
            a1 = first.split("_")[-1].split(".png")[0]
            a2 = second.split("_")[-1].split(".png")[0]
            assert ob1 == ['brown', 'purple', 'cyan', 'c1', 'gray', 'red', 'yellow', 'c2', 'blue'], "locations mixed"
            assert a1 == a2
            action = a1

            if im1 not in transitions:
                transitions[im1] = [(loc1, action, loc2)]
            else:
                transitions[im1].append((loc1, action, loc2))

        train_portion = int(len(transitions)*0.6)
        if train:
            self.transitions = {du1: transitions[du1] for du1 in list(transitions.keys())[:train_portion]}
        else:
            self.transitions = {du1: transitions[du1] for du1 in list(transitions.keys())[train_portion:]}

        self.valid_transitions = []
        self.invalid_actions = []
        for start in self.transitions:
            self.valid_transitions.extend(self.transitions[start])

        print("Loaded %d transitions" % len(self.valid_transitions))

        self.indices = list(self.transitions.keys())
        self.default_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
        return

    def __len__(self):
        return len(self.valid_transitions)

    def __getitem__(self, idx):
        start, act, end = self.valid_transitions[idx]
        act_onehot = torch.zeros(6)
        act_onehot[self.action_list[act]] = 1
        return start, act_onehot, end

def collate_fn_trans(batch):
    loc1, loc2, all_act = [], [], []
    for start, act_onehot, end in batch:
        loc1.append(start)
        loc2.append(end)
        all_act.append(act_onehot.unsqueeze(0))

    loc1 = torch.cat(loc1)
    loc2 = torch.cat(loc2)
    all_act = torch.cat(all_act)

    return loc1, loc2, all_act

def read_scene_json(json_file_dir):
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
    locations = torch.from_numpy(np.array(locations)).float().flatten().unsqueeze(0)
    return bboxes, locations, objects, du["image_filename"]

class Loc2loc(nn.Module):
    def __init__(self):
        super(Loc2loc, self).__init__()


        self.fc = nn.Sequential(
            nn.Linear(24, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 9),
        )


    def forward(self, loc, act):
        return self.fc(torch.cat([loc, act], dim=1))

def find_obj_changed(l1, l2):
    target = []
    for du1 in range(l1.size(0)):
        found = False
        for du2 in range(0, l1.size(1), 2):
            if not found:
                if l1[du1, du2] != l2[du1, du2] or l1[du1, du2+1] != l2[du1, du2+1]:
                    target.append(du2//2)
                    found = True
            else:
                assert l1[du1, du2] == l2[du1, du2] and l1[du1, du2+1] == l2[du1, du2+1]
    return torch.tensor(target).long().to(l1.device)

def eval_procedure(model_, data_iter, device="cuda"):
    total = 0.0
    total2 = 0
    for idx, train_batch in enumerate(data_iter):
        with torch.no_grad():
            loc1, loc2, act = [tensor.to(device) for tensor in train_batch]
            tar = find_obj_changed(loc1, loc2)
            pred = model_(loc1, act)
            acc = torch.sum((torch.argmax(pred, dim=1) == tar))
            total += acc.item()
            total2 += loc2.size(0)
    print(total/total2)
    return total/total2

def train():
    device = "cuda"
    train_data = PBWtransition()
    train_iterator = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=collate_fn_trans)
    val_iterator = DataLoader(PBWtransition(train=False), batch_size=64, shuffle=True, collate_fn=collate_fn_trans)

    model = Loc2loc()
    model.to(device)
    ce_loss = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=2.5e-4, weight_decay=1e-4)
    nb_epc = 500
    for epc in range(nb_epc):
        for idx, train_batch in enumerate(train_iterator):
            loc1, loc2, act = [tensor.to(device) for tensor in train_batch]
            tar = find_obj_changed(loc1, loc2)
            optim.zero_grad()
            pred = model(loc1, act)
            loss = ce_loss(pred, tar)
            loss.backward()
            optim.step()
        eval_procedure(model, val_iterator)

if __name__ == '__main__':
    train()

