from data_loader import PBW_3D
from models import LocationBasedGenerator
from planner import find_top
import torch
import sys
device = "cuda:0"

def test_sg_pred():

    test_data = PBW_3D(train_size=1.0)
    sg2behind = test_data.json2im["sg2behind"]

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
        base_true = []
        for rel in sg:
            if "0" not in rel[2] and "1" not in rel[2] and rel[1] != "left":
                no_base_true.append(rel)
            else:
                base_true.append(rel)

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

        ob2base = {}
        base2ob = {g: [] for g in ["10", "11", "12"]}
        middle_obj = []
        for ob in b_names1:
            if ob in b_names2:
                middle_obj.append(ob)
                ob2base[ob] = "11"
                base2ob["11"].append(ob)
        right_obj = []
        for ob in b_names1:
            if ob not in b_names2:
                right_obj.append(ob)
                ob2base[ob] = "12"
                base2ob["12"].append(ob)
        left_obj = []
        for ob in b_names2:
            if ob not in b_names1:
                left_obj.append(ob)
                ob2base[ob] = "10"
                base2ob["10"].append(ob)

        base2ob_front = {rel[2]: rel[0] for rel in base_true}
        for base in base2ob:
            if len(base2ob[base]) > 0:
                b2 = "0"+base[1]
                if b2 in base2ob_front:
                    res = None
                    for b1 in base2ob[base]:
                        res = b1
                        for p in pred_all:
                            if b1 == p[0] and p[1] == "up":
                                res = None
                                break
                        if res is not None:
                            break
                    assert res is not None
                    pred_all.append([base2ob_front[b2], "front", res])


        if test_data.hash_sg(no_base_true) == test_data.hash_sg(pred_all):
            acc2 += 1

        acc += 1
        total += 1
        print(acc, total, acc2, acc/total, acc2/total)

def find_closest(up_rel, ob_start):
    for rel in up_rel:
        if rel[2] == ob_start:
            return rel[0]

def action_model(sg_from):
    all_bases = ["00", "01", "02", "10", "11", "12"]
    present_bases = [rel[2] for rel in sg_from if rel[2] in all_bases]
    up_rel = [rel for rel in sg_from if rel[1] == "up"]
    neighboring_sg = []
    for b1 in present_bases[:]:
        for b2 in all_bases[:]:
            if b1 != b2:
                o1, r1 = find_top(up_rel, b1)
                o2, r2 = find_top(up_rel, b2)

                sg_neighbor = sg_from[:]
                sg_neighbor.remove(r1)
                for rel in sg_neighbor:
                    if rel[0] == o1 or rel[2] == o1:
                        sg_neighbor.remove(rel)

                sg_neighbor.append([o1, "up", o2])
                if r2 is None:
                    if b2[0] == "0":
                        behind_base = "1"+b2[1]
                        if behind_base in present_bases:
                            behind_obj = find_closest(up_rel, behind_base)
                            sg_neighbor.append([o1, "front", behind_obj])
                    elif b2[0] == "1":
                        front_base = "0" + b2[1]
                        if front_base in present_bases:
                            front_obj = find_closest(up_rel, front_base)
                            sg_neighbor.append([front_obj, "front", o1])
                neighboring_sg.append([sg_neighbor, (b1, b2, o1, o2)])
    return neighboring_sg

def plan():
    test_data = PBW_3D(train_size=1.0)
    sg2behind = test_data.json2im["sg2behind"]

    for sample in test_data:
        sg, obj_names, front_masks, behind_masks, behind_objects, \
        sg_n1, sg_n2, im_name, front_masks_name, behind_masks_name = sample
        print(im_name, sg)
        sg_neighbors = action_model(sg)
        for sg_n, info in sg_neighbors:
            try:
                _, _, im_n = sg2behind[test_data.hash_sg(sg_n)]
            except KeyError:
                print("wrong", sg_n)
                print(info)
                sys.exit()
            print("   ", im_n)
        break

plan()