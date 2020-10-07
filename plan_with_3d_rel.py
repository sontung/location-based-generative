from data_loader import PBW_3D
from models import LocationBasedGenerator
from utils import kmeans
import torch
import sys
device = "cuda:0"

test_data = PBW_3D()
sg2behind = test_data.json2im["sg2behind"]
print(len(sg2behind))

model = LocationBasedGenerator()
model.to(device)
model.load_state_dict(torch.load("pre_models/model-20200726-041935", map_location=device))
acc = 0
acc2 = 0
total = 0.0


for test_sample in test_data:
    sg, obj_names, front_masks, behind_masks, behind_objects,\
    sg_n1, sg_n2, im_name, front_masks_name, behind_masks_name = test_sample

    front_masks = front_masks.to(device)

    pred_front_masks, front_trans_vec = model.return_sg(front_masks, [front_masks_name], if_return_trans_vec=True)
    pred_front_masks = pred_front_masks[0]
    for pred in pred_front_masks:
        if pred not in sg:
            continue

    b_masks1, b_names1, _ = sg2behind[test_data.hash_sg(sg_n1)]
    if b_masks1 != []:
        b_masks1 = b_masks1.to(device)
        pred_behind_masks1, b_trans_vec1 = model.return_sg(b_masks1, [b_names1], if_return_trans_vec=True)
        pred_behind_masks1 = pred_behind_masks1[0]
        for pred in pred_behind_masks1:
            if pred not in sg:
                continue
    else:
        pred_behind_masks1 = []
        b_trans_vec1 = []

    b_masks2, b_names2, _ = sg2behind[test_data.hash_sg(sg_n2)]
    if b_masks2 != []:
        b_masks2 = b_masks2.to(device)
        pred_behind_masks2, b_trans_vec2 = model.return_sg(b_masks2, [b_names2], if_return_trans_vec=True)
        pred_behind_masks2 = pred_behind_masks2[0]
        for pred in pred_behind_masks2:
            if pred not in sg:
                continue
    else:
        pred_behind_masks2 = []
        b_trans_vec2= []

    pred_all = []
    total_list = pred_behind_masks1 + pred_behind_masks2 + pred_front_masks

    for pred in total_list:
        if pred not in pred_all and len(pred) > 0:
            pred_all.append(pred)

    no_base_true = []
    for rel in sg:
        if "0" not in rel[2] and "1" not in rel[2] and rel[1] != "left":
            no_base_true.append(rel)

    # add front rel
    all_name = front_masks_name + b_names1 + b_names2
    all_b_names = b_names1 + b_names2
    trans = [front_trans_vec]
    if b_trans_vec1 != []:
        trans.append(b_trans_vec1)
    if b_trans_vec2 != []:
        trans.append(b_trans_vec2)

    # remove left
    for rel in pred_all[:]:
        if rel[1] == "left":
            pred_all.remove(rel)

    all_trans_vec = torch.cat(trans, dim=1).squeeze()
    assigns = kmeans(all_trans_vec[:, 1], 3)
    name2assign = {all_name[du]: assigns[du] for du in range(len(all_name))}

    for f_name in front_masks_name:
        for b_name in all_b_names:
            if name2assign[f_name] == name2assign[b_name]:
                pred_all.append([f_name, "front", b_name])

    if test_data.hash_sg(no_base_true) == test_data.hash_sg(pred_all):
        acc2 += 1
    else:
        print(sorted(no_base_true))
        print(sorted(pred_all))
        print(front_masks_name, b_names1, b_names2)
        # all_trans_vec = torch.cat(trans, dim=1).squeeze()
        # print(all_trans_vec)
        # print(kmeans(all_trans_vec[:, 0], 3))
        # print(kmeans(all_trans_vec[:, 1], 3))
        #
        sys.exit()

    acc += 1
    total += 1
    print(acc, total, acc2, acc/total, acc2/total)
