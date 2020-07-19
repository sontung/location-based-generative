import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from torchvision.utils import make_grid

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
    relationships.append([bottoms[0][0], "left", bottoms[1][0]])
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
