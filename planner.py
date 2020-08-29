import utils
import os
import random
import sys
import torchvision
import torch
import pickle
from PIL import Image
from queue import Queue
from models import LocationBasedGenerator
from data_loader import PBW_Planning_only
from utils import show2


class Node:
    def __init__(self, sg, ob_names, if_goal=False):
        self.sg = sg
        self.key = hash_sg(sg, ob_names)

        self.visited = False
        self.parent = None
        self.act = None
        self.goal = if_goal

    def get_key(self):
        return self.key

    def get_sg(self):
        return self.sg

    def __eq__(self, other):
        if other.goal:
            for i in range(len(self.key)):
                if other.key[i] > 0 and other.key[i] != self.key[i]:
                    return False
            return True
        else:
            return self.__class__ == other.__class__ and self.key == other.key


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


def action_model(sg_from, action):
    assert action in ["12", "13", "21", "23", "31", "32"]
    base_objs = ['brown', 'purple', 'cyan']
    relations_from = sg_from[:]
    block_from = int(action[0])-1
    block_to = int(action[1])-1

    # check valid action
    up_rel = [rel for rel in relations_from if rel[1] == "up"]
    ob_to = [rel[2] for rel in up_rel]
    if base_objs[block_from] not in ob_to:
        return None

    # modify "from" block
    top_block_from, to_be_removed_rel = find_top(up_rel, base_objs[block_from])
    relations_from.remove(to_be_removed_rel)

    # modify "to" block
    top_block_to, _ = find_top(up_rel, base_objs[block_to])
    relations_from.append([top_block_from, "up", top_block_to])
    assert top_block_from != top_block_to

    return relations_from


def possible_next_states(current_state):
    predefined_actions = ["12", "13", "21", "23", "31", "32"]
    res = []
    for action in predefined_actions:
        next_state = action_model(current_state, action)
        if next_state is not None:
            res.append((next_state, action))
    return res


def hash_sg(relationships, ob_names):
    """
    hash into unique ID
    :param relationships: [['brown', 'left', 'purple'] , ['yellow', 'up', 'yellow']]
    :param ob_names:
    :return:
    """
    a_key = [0]*64
    pred2id = {"none": 0, "left": 1, "up": 2}
    predefined_objects1 = ob_names[:]
    predefined_objects2 = ob_names[:]
    pair2pred = {}
    for rel in relationships:
        if rel[1] != "__in_image__":
            pair2pred[(rel[0], rel[2])] = pred2id[rel[1]]

    idx = 0
    for ob1 in predefined_objects1:
        for ob2 in predefined_objects2:
            if (ob1, ob2) in pair2pred:
                a_key[idx] = pair2pred[(ob1, ob2)]
            idx += 1
    return tuple(a_key)


def inv_hash_sg(a_key, ob_names):
    predefined_objects1 = ob_names[:]
    predefined_objects2 = ob_names[:]
    id2pred = {0: "none", 1: "left", 2: "up"}

    sg = []
    idx = 0
    for ob1 in predefined_objects1:
        for ob2 in predefined_objects2:
            if a_key[idx] > 0:
                sg.append([ob1, id2pred[a_key[idx]], ob2])
            idx += 1
    return sg


def command_by_sg_sg(start_sg, end_sg, ob_names):
    # bfs
    Q = Queue()
    start_v = Node(start_sg, ob_names)
    goal_state = Node(end_sg, ob_names, if_goal=True)
    all_nodes = {start_v.get_key(): start_v, goal_state.get_key(): goal_state}
    start_v.visited = True
    Q.put(start_v)
    while not Q.empty():
        v = Q.get()
        if v == goal_state:
            print("search is completed")
            goal_state = v
            break
        for w, act in possible_next_states(v.get_sg()):
            w_key = hash_sg(w, ob_names)
            if w_key not in all_nodes:
                all_nodes[w_key] = Node(w, ob_names)
            if not all_nodes[w_key].visited:
                all_nodes[w_key].visited = True
                all_nodes[w_key].parent = v
                all_nodes[w_key].act = act
                Q.put(all_nodes[w_key])

    traces = []
    actions = []
    while True:
        traces.insert(0, goal_state.get_key())
        actions.insert(0, goal_state.act)
        if goal_state == start_v:
            break
        goal_state = goal_state.parent
    return traces, actions


def visualize_plan(im_list, perrow=9, if_save=False):
    im_tensors = []
    transform = torchvision.transforms.ToTensor()
    for idx, im_name in enumerate(im_list):
        im = Image.open(im_name).convert('RGB')
        im_tensors.append(transform(im).unsqueeze(0))
        if if_save:
            im.save("figures/%d.png" % idx)
    im_tensors = torch.cat(im_tensors, dim=0)
    show2([im_tensors], "solution", perrow)


def command_by_im_im(model_, device="cuda", name="blocks-5-3-2520-planning", domain_task="blocks-5-3",
                     nb_objects=8):
    _ = PBW_Planning_only(root_dir="/home/sontung/thesis/photorealistic-blocksworld/%s" % domain_task, nb_samples=-1)
    ob_names = ['brown', 'purple', 'cyan', 'gray', 'blue', 'red', 'green', 'yellow'][:nb_objects]
    ob_names = [ob_names, ob_names]
    print("Loading precomputed json2im:", "data/%s" % name)
    with open("data/%s" % name, 'rb') as f:
        js2mask = pickle.load(f)

    sg2im = {}
    for dm3 in js2mask:
        k = hash_sg(js2mask[dm3][1], ob_names[0])
        sg2im[k] = dm3.replace("json", "png").replace("scene", "image")

    keys = list(js2mask.keys())
    all_plans = []
    for _ in range(10):
        before_name = random.choice(keys)
        after_name = random.choice(keys)
        start = js2mask[before_name][0].to(device)
        ob_names1 = js2mask[before_name][-1]
        end = js2mask[after_name][0].to(device)
        ob_names2 = js2mask[after_name][-1]

        assert ob_names1 == ob_names2 == ob_names[0]

        print("planning from %s to %s" % (before_name.split("/")[-1], after_name.split("/")[-1]))
        input_images = torch.cat([start, end], dim=0).to(device)
        with torch.no_grad():
            sg_from, sg_to = model_.return_sg(input_images, ob_names)
        print("start", sg_from)
        print("end", sg_to)
        tr, ac = command_by_sg_sg(sg_from, sg_to, ob_names[0])
        print(ac)

        image_list = []
        for t in tr:
            image_list.append(sg2im[t])
            if len(image_list) == 5:
                all_plans.extend(image_list)
                break
    visualize_plan(all_plans[:15], perrow=5)


def command_by_sg_sg_partial(name="blocks-5-3-2520-planning"):
    sg_from = [['red', 'up', 'yellow'], ['yellow', 'up', 'purple'], ['green', 'up', 'blue'],
               ['blue', 'up', 'gray'], ['gray', 'up', 'cyan'], ['brown', 'left', 'purple'],
               ['purple', 'left', 'cyan']]
    sg_to = [['yellow', 'up', 'gray'], ['blue', 'up', 'red'], ["green", "up", "cyan"]]
    ob_names = ['brown', 'purple', 'cyan', 'gray', 'blue', 'red', 'green', 'yellow']
    tr, ac = command_by_sg_sg(sg_from, sg_to, ob_names)

    with open("data/%s" % name, 'rb') as f:
        js2mask = pickle.load(f)

    sg2im = {}
    for dm3 in js2mask:
        k = hash_sg(js2mask[dm3][1], ob_names)
        sg2im[k] = dm3.replace("json", "png").replace("scene", "image")

    image_list = []
    for t in tr:
        try:
            image_list.append(sg2im[t])
        except KeyError:
            print(inv_hash_sg(t, ob_names))
    visualize_plan(image_list)
    return


if __name__ == '__main__':
    device = "cuda"
    sae = LocationBasedGenerator()
    sae.to(device)
    sae.load_state_dict(torch.load("pre_models/model-20200726-041935", map_location=device))

    command_by_im_im(sae)

