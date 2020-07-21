import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import torchvision
from torchvision.utils import make_grid
from PIL import Image


def compute_grad(model_):
    res = 0
    for param in model_.parameters():
        if param.requires_grad:
            if param.grad is not None:
                res += abs(torch.sum(torch.abs(param.grad)).item())
    return res

def show2(im_, name, nrow):
    import logging

    logger = logging.getLogger()
    old_level = logger.level
    logger.setLevel(100)

    fig_ = plt.figure(figsize=(15, 15))
    for du3 in range(1, len(im_)+1):
        plt.subplot(1, len(im_), du3)
        plt.axis("off")
        plt.imshow(np.transpose(make_grid(im_[du3-1], padding=5, normalize=False, pad_value=50, nrow=nrow),
                                (1, 2, 0)))

    plt.axis("off")
    # plt.title("black: no action, red: 1-3, yellow: 3-1, green: 1-2, blue: 2-3, pink: 3-2, brown: 2-1")
    plt.savefig(name, transparent=True, bbox_inches='tight')
    plt.close(fig_)
    logger.setLevel(old_level)

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
            location_dict[a_key] = [(object_id, locations[idx][1], locations[idx][0])]
        else:
            location_dict[a_key].append((object_id, locations[idx][1], locations[idx][0]))
        objects.append(object_id)
    relationships = []

    # decide up relation
    bottoms = []
    for du3 in location_dict:
        location = sorted(location_dict[du3], key=lambda x: x[1])
        bottoms.append(location[0])
        while len(location) > 1:
            o1 = location.pop()[0]
            o2 = location[-1][0]
            relationships.append([o1, "up", o2])

    # decide left relation
    bottoms = sorted(bottoms, key=lambda x: x[2])

    if len(bottoms) > 1:
        relationships.append([bottoms[0][0], "left", bottoms[1][0]])
    if len(bottoms) > 2:
        relationships.append([bottoms[1][0], "left", bottoms[2][0]])
    if if_return_assigns:
        return relationships, k_means_assign
    return relationships

def kmeans(data_):
    """
    assign each object one of 3 block IDs based on the x coord
    :param data_:
    :return:
    """
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

def name2sg(name):
    name = name.replace(".ppm", "")
    numb2color = {
        0: "red",
        1: "green",
        2: "brown",
        3: "blue",
        4: "pink"
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
    if bottoms[2] != 0 and bottoms[1] != 0:
        relationships.append([bottoms[1], "left", bottoms[2]])
    if bottoms[0] != 0 and bottoms[1] != 0:
        relationships.append([bottoms[0], "left", bottoms[1]])
    return relationships

def return_default_mat(im_tensor):
    from data_loader import construct_weight_map

    im_np = im_tensor.numpy()[0, :, :]
    a = np.where(im_np != 0)
    bbox_int = np.min(a[1]), np.min(a[0]), np.max(a[1]), np.max(a[0])
    default_inp = torch.zeros_like(im_tensor)
    idc1 = range(bbox_int[0], bbox_int[2] + 1)
    idc2 = range(len(idc1))
    for j_, y_ in enumerate(range(bbox_int[1], bbox_int[3] + 1)):
        default_inp[:, j_ + 128 - int(-bbox_int[1] + bbox_int[3] + 1), idc2] = im_tensor[:, y_, idc1]

    weight = construct_weight_map(bbox_int)

    return default_inp, weight

def read_seg_masks(im_dir="/home/sontung/Downloads/5objs_seg/z.seg2153_s2_423_s0_0_s1_1.ppm"):

    im_pil = Image.open(im_dir).convert('RGB')
    transform = torchvision.transforms.ToTensor()
    im_mat = transform(im_pil)
    cl2name = {
       (66, 0, 192): 'blue', 
       (194, 0, 192): 'pink',
       (194, 128, 64): 'brown', 
       (66, 128, 64): 'green', 
       (194, 0, 64): 'red',
       (64, 128, 194): 'navy'
       }
    ob_names = ['blue', 'pink', 'brown', 'green', 'red', 'navy']
    name2mask = {dm3: torch.zeros(3, 128, 128) for dm3 in ob_names}

    existing_colors = []
    for i in range(im_mat.size(1)):
        for j in range(im_mat.size(1)):
            color = tuple([
                int(im_mat[0, i, j].item()*255),
                int(im_mat[1, i, j].item()*255),
                int(im_mat[2, i, j].item()*255),
                ])
            if torch.sum(im_mat[:, i, j]).item() == 0:
                continue
            if cl2name[color] not in existing_colors:
                existing_colors.append(cl2name[color])
            name2mask[cl2name[color]][:, i, j] = im_mat[:, i, j]
    for dm55 in ob_names:
        if dm55 not in existing_colors:
            assert dm55 == "navy"
            ob_names.remove(dm55)
            del name2mask["navy"]

    masks = torch.cat([name2mask[dm4].unsqueeze(0) for dm4 in ob_names], dim=0)
    def_wei = [return_default_mat(name2mask[dm4]) for dm4 in ob_names]
    def_mat = torch.cat([dm4[0].unsqueeze(0) for dm4 in def_wei], dim=0)
    wei_mat = torch.cat([dm4[1].unsqueeze(0) for dm4 in def_wei], dim=0)
    # show2([masks, def_mat, wei_mat], "masks_test", 5)

    return masks, def_mat, wei_mat, ob_names, name2sg(im_dir.split("/")[-1])


if __name__ == '__main__':
    read_seg_masks()
