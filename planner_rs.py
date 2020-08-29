import random
import sys
import torchvision
import torch
import pickle
from PIL import Image
from queue import Queue
from models import LocationBasedGenerator
from utils import show2, read_seg_masks, name2sg
from os import listdir
from os.path import isfile, join
from planner import visualize_plan

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
    base_objs = ['0', '1', '2']
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
    a_key = [0]*len(ob_names)*len(ob_names)
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
    return tuple(sorted(sg))


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

def add_im_bases(name):
    name = name.replace(".ppm", "")
    numb2color = {
        0: "red",
        1: "green",
        2: "brown",
        3: "blue",
        4: "pink",
        5: "navy",
        6: "pink2",
    }
    infor = name.split("_")[1:]
    bottoms = [0, 0, 0]
    relationships = []

    for dm in range(len(infor)):
        if "s" in infor[dm]:
            stack = infor[dm+1]
            if len(stack) == 0:
                continue
            bottoms[int(infor[dm][1])] = numb2color[int(stack[0])]
            for dm2 in range(len(stack)-1):
                o1 = stack[dm2]
                o2 = stack[dm2+1]
                relationships.append([numb2color[int(o2)], "up", numb2color[int(o1)]])
    return [[bot, "up", str(im_base)] for im_base, bot in enumerate(bottoms) if bot != 0]

def remove_left_rel(sg):
    res = []
    for u in sg:
        if u[1] != "left":
            res.append(u)
    return res

def build_sg2im(root_dir, obnames):
    sg2im = {}
    good_list = [f for f in listdir(root_dir) if isfile(join(root_dir, f))]
    for f in good_list:
        sg = name2sg(f)
        sg.extend(add_im_bases(f))
        sg = remove_left_rel(sg)
        sg2im[hash_sg(sg, obnames)] = "%s/%s" % (root_dir, f)

    return sg2im

def interpret_action(sg_from, action):
    assert action in ["12", "13", "21", "23", "31", "32"]
    base_objs = ['0', '1', '2']
    relations_from = sg_from[:]
    block_from = int(action[0])-1
    block_to = int(action[1])-1

    # check valid action
    up_rel = [rel for rel in relations_from if rel[1] == "up"]

    # modify "from" block
    top_block_from, to_be_removed_rel = find_top(up_rel, base_objs[block_from])

    # modify "to" block
    top_block_to, _ = find_top(up_rel, base_objs[block_to])

    assert top_block_from != top_block_to
    return [top_block_from, "up", top_block_to]

def pretty_print_actions(seq):
    print("[", end="")
    for a in seq:
        a = a.replace("0", "stack0")
        a = a.replace("1", "stack1")
        a = a.replace("2", "stack2")
        print("\"%s\", " % a, end="")
    print("]\n")

def plan():

    device = "cpu"
    sae = LocationBasedGenerator()
    sae.to(device)
    sae.load_state_dict(torch.load("pre_models/model-sim-20200725-114336", map_location=device))
    sae.eval()

    root_dir = "/home/sontung/Downloads/6objs_seg"
    wrong_list = ['z.seg385_s0_310254_s1__s2_.ppm', 'z.seg2049_s2_502341_s0__s1_.ppm', 'z.seg289_s0_230154_s1__s2_.ppm', 'z.seg1069_s1_253041_s2__s0_.ppm', 'z.seg349_s0_253041_s1__s2_.ppm', 'z.seg255_s0_204351_s1__s2_.ppm', 'z.seg367_s0_302154_s1__s2_.ppm', 'z.seg615_s0_503241_s1__s2_.ppm', 'z.seg1371_s1_520341_s2__s0_.ppm', 'z.seg121_s0_102354_s1__s2_.ppm', 'z.seg1691_s2_203541_s0__s1_.ppm', 'z.seg1867_s2_325041_s0__s1_.ppm', 'z.seg1551_s2_053241_s0__s1_.ppm', 'z.seg45_s0_025341_s1__s2_.ppm', 'z.seg445_s0_342051_s1__s2_.ppm', 'z.seg1590_s2_123045_s0__s1_.ppm', 'z.seg261_s0_205341_s1__s2_.ppm', 'z.seg1909_s2_352041_s0__s1_.ppm', 'z.seg1104_s1_310245_s2__s0_.ppm', 'z.seg990_s1_213045_s2__s0_.ppm', 'z.seg1830_s2_312045_s0__s1_.ppm', 'z.seg381_s0_305241_s1__s2_.ppm', 'z.seg1899_s2_350241_s0__s1_.ppm', 'z.seg1110_s1_312045_s2__s0_.ppm', 'z.seg960_s1_201345_s2__s0_.ppm', 'z.seg1091_s1_302541_s2__s0_.ppm', 'z.seg411_s0_320451_s1__s2_.ppm', 'z.seg750_s1_023145_s2__s0_.ppm', 'z.seg427_s0_325041_s1__s2_.ppm', 'z.seg271_s0_213054_s1__s2_.ppm', 'z.seg39_s0_024351_s1__s2_.ppm', 'z.seg375_s0_304251_s1__s2_.ppm', 'z.seg293_s0_230541_s1__s2_.ppm', 'z.seg1446_s2_013245_s0__s1_.ppm', 'z.seg1853_s2_320541_s0__s1_.ppm', 'z.seg1811_s2_302541_s0__s1_.ppm', 'z.seg1854_s2_321045_s0__s1_.ppm', 'z.seg87_s0_043251_s1__s2_.ppm', 'z.seg1614_s2_132045_s0__s1_.ppm', 'z.seg489_s0_402351_s1__s2_.ppm', 'z.seg459_s0_350241_s1__s2_.ppm', 'z.seg57_s0_032451_s1__s2_.ppm', 'z.seg145_s0_120354_s1__s2_.ppm', 'z.seg175_s0_132054_s1__s2_.ppm', 'z.seg755_s1_023541_s2__s0_.ppm', 'z.seg1133_s1_320541_s2__s0_.ppm', 'z.seg1734_s2_231045_s0__s1_.ppm', 'z.seg1494_s2_032145_s0__s1_.ppm', 'z.seg888_s1_130245_s2__s0_.ppm', 'z.seg1728_s2_230145_s0__s1_.ppm', 'z.seg295_s0_231054_s1__s2_.ppm', 'z.seg1821_s2_305241_s0__s1_.ppm', 'z.seg1013_s1_230541_s2__s0_.ppm', 'z.seg768_s1_031245_s2__s0_.ppm', 'z.seg720_s1_012345_s2__s0_.ppm', 'z.seg1686_s2_203145_s0__s1_.ppm', 'z.seg1101_s1_305241_s2__s0_.ppm', 'z.seg1086_s1_302145_s2__s0_.ppm', 'z.seg541_s0_423051_s1__s2_.ppm', 'z.seg864_s1_120345_s2__s0_.ppm', 'z.seg1584_s2_120345_s0__s1_.ppm', 'z.seg1329_s1_502341_s2__s0_.ppm', 'z.seg63_s0_034251_s1__s2_.ppm', 'z.seg413_s0_320541_s1__s2_.ppm', 'z.seg49_s0_031254_s1__s2_.ppm', 'z.seg1806_s2_302145_s0__s1_.ppm', 'z.seg421_s0_324051_s1__s2_.ppm', 'z.seg1509_s2_035241_s0__s1_.ppm', 'z.seg1848_s2_320145_s0__s1_.ppm', 'z.seg435_s0_340251_s1__s2_.ppm', 'z.seg369_s0_302451_s1__s2_.ppm', 'z.seg661_s0_523041_s1__s2_.ppm', 'z.seg779_s1_032541_s2__s0_.ppm', 'z.seg409_s0_320154_s1__s2_.ppm', 'z.seg555_s0_430251_s1__s2_.ppm', 'z.seg265_s0_210354_s1__s2_.ppm', 'z.seg371_s0_302541_s1__s2_.ppm', 'z.seg685_s0_532041_s1__s2_.ppm', 'z.seg7_s0_013254_s1__s2_.ppm', 'z.seg495_s0_403251_s1__s2_.ppm', 'z.seg846_s1_103245_s2__s0_.ppm', 'z.seg1747_s2_235041_s0__s1_.ppm', 'z.seg1701_s2_205341_s0__s1_.ppm', 'z.seg241_s0_201354_s1__s2_.ppm', 'z.seg1395_s1_530241_s2__s0_.ppm', 'z.seg1499_s2_032541_s0__s1_.ppm', 'z.seg1704_s2_210345_s0__s1_.ppm', 'z.seg169_s0_130254_s1__s2_.ppm', 'z.seg1566_s2_103245_s0__s1_.ppm', 'z.seg2101_s2_523041_s0__s1_.ppm', 'z.seg825_s1_052341_s2__s0_.ppm', 'z.seg247_s0_203154_s1__s2_.ppm', 'z.seg1608_s2_130245_s0__s1_.ppm', 'z.seg69_s0_035241_s1__s2_.ppm', 'z.seg315_s0_240351_s1__s2_.ppm', 'z.seg609_s0_502341_s1__s2_.ppm', 'z.seg301_s0_234051_s1__s2_.ppm', 'z.seg1789_s2_253041_s0__s1_.ppm', 'z.seg25_s0_021354_s1__s2_.ppm', 'z.seg984_s1_210345_s2__s0_.ppm', 'z.seg981_s1_205341_s2__s0_.ppm', 'z.seg1335_s1_503241_s2__s0_.ppm', 'z.seg31_s0_023154_s1__s2_.ppm', 'z.seg1440_s2_012345_s0__s1_.ppm', 'z.seg1710_s2_213045_s0__s1_.ppm', 'z.seg33_s0_023451_s1__s2_.ppm', 'z.seg1014_s1_231045_s2__s0_.ppm', 'z.seg971_s1_203541_s2__s0_.ppm', 'z.seg81_s0_042351_s1__s2_.ppm', 'z.seg291_s0_230451_s1__s2_.ppm', 'z.seg1059_s1_250341_s2__s0_.ppm', 'z.seg391_s0_312054_s1__s2_.ppm', 'z.seg870_s1_123045_s2__s0_.ppm', 'z.seg1381_s1_523041_s2__s0_.ppm', 'z.seg1733_s2_230541_s0__s1_.ppm', 'z.seg105_s0_052341_s1__s2_.ppm', 'z.seg2115_s2_530241_s0__s1_.ppm', 'z.seg1488_s2_031245_s0__s1_.ppm', 'z.seg325_s0_243051_s1__s2_.ppm', 'z.seg1824_s2_310245_s0__s1_.ppm', 'z.seg1800_s2_301245_s0__s1_.ppm', 'z.seg831_s1_053241_s2__s0_.ppm', 'z.seg2091_s2_520341_s0__s1_.ppm', 'z.seg1_s0_012354_s1__s2_.ppm', 'z.seg1485_s2_025341_s0__s1_.ppm', 'z.seg1008_s1_230145_s2__s0_.ppm', 'z.seg35_s0_023541_s1__s2_.ppm', 'z.seg966_s1_203145_s2__s0_.ppm', 'z.seg1560_s2_102345_s0__s1_.ppm', 'z.seg2125_s2_532041_s0__s1_.ppm', 'z.seg2055_s2_503241_s0__s1_.ppm', 'z.seg1405_s1_532041_s2__s0_.ppm', 'z.seg531_s0_420351_s1__s2_.ppm', 'z.seg1680_s2_201345_s0__s1_.ppm', 'z.seg127_s0_103254_s1__s2_.ppm', 'z.seg339_s0_250341_s1__s2_.ppm', 'z.seg726_s1_013245_s2__s0_.ppm', 'z.seg1080_s1_301245_s2__s0_.ppm', 'z.seg1470_s2_023145_s0__s1_.ppm', 'z.seg1027_s1_235041_s2__s0_.ppm', 'z.seg59_s0_032541_s1__s2_.ppm', 'z.seg774_s1_032145_s2__s0_.ppm', 'z.seg151_s0_123054_s1__s2_.ppm', 'z.seg1189_s1_352041_s2__s0_.ppm', 'z.seg840_s1_102345_s2__s0_.ppm', 'z.seg111_s0_053241_s1__s2_.ppm', 'z.seg1545_s2_052341_s0__s1_.ppm', 'z.seg1779_s2_250341_s0__s1_.ppm', 'z.seg307_s0_235041_s1__s2_.ppm', 'z.seg651_s0_520341_s1__s2_.ppm', 'z.seg894_s1_132045_s2__s0_.ppm', 'z.seg565_s0_432051_s1__s2_.ppm', 'z.seg1147_s1_325041_s2__s0_.ppm', 'z.seg249_s0_203451_s1__s2_.ppm', 'z.seg415_s0_321054_s1__s2_.ppm', 'z.seg1128_s1_320145_s2__s0_.ppm', 'z.seg55_s0_032154_s1__s2_.ppm', 'z.seg1464_s2_021345_s0__s1_.ppm', 'z.seg1475_s2_023541_s0__s1_.ppm', 'z.seg469_s0_352041_s1__s2_.ppm', 'z.seg1179_s1_350241_s2__s0_.ppm', 'z.seg765_s1_025341_s2__s0_.ppm', 'z.seg361_s0_301254_s1__s2_.ppm', 'z.seg1134_s1_321045_s2__s0_.ppm', 'z.seg744_s1_021345_s2__s0_.ppm', 'z.seg789_s1_035241_s2__s0_.ppm', 'z.seg251_s0_203541_s1__s2_.ppm', 'z.seg675_s0_530241_s1__s2_.ppm']

    good_list = [join(root_dir, f) for f in listdir(root_dir) if isfile(join(root_dir, f)) and f not in wrong_list]

    while True:
        im_file1 = "/home/sontung/Downloads/6objs_seg/z.seg9958_s1_0432_s2__s0_51.ppm" # random.choice(good_list)
        im_file2 = "/home/sontung/Downloads/6objs_seg/z.seg15278_s1_203_s2_415_s0_.ppm" # random.choice(good_list)

        # im_file1 = random.choice(good_list)
        # im_file2 = random.choice(good_list)


        masks, def_mat, wei_mat, ob_names, sgtrue1, im_names = read_seg_masks(im_file1)
        sg1 = sae.return_sg(masks.unsqueeze(0), [ob_names])[0]
        masks, _, _, ob_names, sgtrue2, _ = read_seg_masks(im_file2)
        sg2 = sae.return_sg(masks.unsqueeze(0), [ob_names])[0]

        try:
            assert sg1 == sgtrue1 and sg2 == sgtrue2
        except AssertionError:
            print(sg1)
            print(sgtrue1)
            print(sg2)
            print(sgtrue2)
            print("failed to capture, please re run")
            sys.exit()

        ob_names.extend(["0", "1", "2"])  # imaginary bases
        im_base1 = add_im_bases(im_file1.split("/")[-1])
        im_base2 = add_im_bases(im_file2.split("/")[-1])

        print(im_file1, im_file2)
        sg1.extend(im_base1)
        sg2.extend(im_base2)

        # remove left
        sg1 = remove_left_rel(sg1)
        sg2 = remove_left_rel(sg2)

        # override
        # sg2 = [['blue', 'up', 'red'], ['green', 'up', 'pink'],
        #         ['pink', 'up', '2']]

        # if ["navy", "up", "0"] not in sg1 and ["navy", "up", "1"] not in sg1 and ["navy", "up", "2"] not in sg1:
        #     continue
        print(sg1, "\n", sg2)


        sg2im = build_sg2im(root_dir, ob_names)
        tr, ac = command_by_sg_sg(sg1, sg2, ob_names)

        image_list = []
        print(ac)

        readable_actions = []
        for idx, action in enumerate(ac):
            if action is not None:
                prev_sg = inv_hash_sg(tr[idx-1], ob_names)
                act2 = interpret_action(prev_sg, action)
                readable_actions.append("pick %s and place on %s" % (act2[0], act2[2]))
        pretty_print_actions(readable_actions)

        for t in tr:
            try:
                image_list.append(sg2im[t])
            except KeyError:
                print("failed to visualize, please re run")
                sys.exit()
        print("last image", image_list[-1])
        visualize_plan(image_list, if_save=True)
        break


plan()