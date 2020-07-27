import torch
import random
import cairo
import numpy as np
import math
import os
import torchvision
import skimage.draw
import skimage.io
from models import LocationBasedGenerator
from PIL import Image
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from utils import return_default_mat, show2, compute_iou
from os import listdir
from os.path import isfile, join

def draw_circle(cent, color):
    mat = np.zeros([128, 128, 3], dtype=np.uint8)
    circle = skimage.draw.circle(cent[0], cent[1], 10)
    mat[circle] = color
    return mat

def draw_sq(cent, color, size=7):
    mat = np.zeros([128, 128, 3], dtype=np.uint8)
    square = skimage.draw.polygon([cent[0]-size, cent[0]+size, cent[0]+size, cent[0]-size],
                                  [cent[1]-size, cent[1]-size, cent[1]+size, cent[1]+size])
    mat[square] = color
    return mat

def draw_tri(cent, color, size=7):
    mat = np.zeros([128, 128, 3], dtype=np.uint8)
    square = skimage.draw.polygon([cent[0] - size, cent[0], cent[0] + size],
                                  [cent[1] + size, cent[1] - size, cent[1] + size])
    mat[square] = color
    return mat

name2color = {
    'blue': [66, 0, 192],
    'pink': [194, 0, 192],
    'brown': [194, 128, 64],
    'green': [66, 128, 64],
    'red': [194, 0, 64],
    'navy': [66, 128, 192],
    'pink2': [194, 128, 192],
    'c-0': [3, 153, 183], 'c-1': [29, 81, 34], 'c-2': [143, 64, 188], 'c-3': [242, 219, 176],
    'c-4': [196, 218, 69], 'c-5': [165, 67, 114], 'c-6': [187, 187, 253]
}

shape2func = {
    "circle": draw_circle,
    "square": draw_sq,
    "tri": draw_tri
}

def create_data():
    nb_samples = 1000
    for color in name2color:
        for shape in shape2func:
            print(color, shape)
            save_dir = "data/fine-scale/%s-%s" % (color, shape)
            for i in range(nb_samples):
                cent = [random.randint(14, 114), random.randint(14, 114)]
                im = shape2func[shape](cent, name2color[color])
                skimage.io.imsave("%s/mat-%d.png" % (save_dir, i), im)

def analyze():
    device = "cuda"
    sae = LocationBasedGenerator()
    sae.to(device)
    sae.load_state_dict(torch.load("pre_models/model-sim-20200725-114336", map_location=device))
    
    color_data = {}
    shape_data = {}
    for color in name2color:
        for shape in shape2func:
            root_dir = "data/fine-scale/%s-%s" % (color, shape)

            all_files = [join(root_dir, f) for f in listdir(root_dir) if isfile(join(root_dir, f))]
            data = []
            trans = torchvision.transforms.ToTensor()
            for a_file in all_files:
                img_pil = Image.open(a_file).convert('RGB')
                img = trans(img_pil).unsqueeze(0)
                data.append(img)
            data = torch.cat(data, dim=0)
            data_loader =  DataLoader(data, batch_size=64, shuffle=False)
            ious = []
            for batch in data_loader:
                def_mat = torch.cat([return_default_mat(batch[i])[0].unsqueeze(0)
                                     for i in range(batch.size(0))], dim=0).to(device)
                batch = batch.to(device)

                with torch.no_grad():
                    pred = sae.infer(batch, def_mat)
                ious.extend(compute_iou(pred, batch)[0])
                # show2([pred.cpu(), batch.cpu(), def_mat.cpu()], "test", 4)
            print(root_dir, np.mean(ious))
            
            if color not in color_data:
                color_data[color] = [np.mean(ious)]
            else:
                color_data[color].append(np.mean(ious))
            
            if shape not in shape_data:
                shape_data[shape] = [np.mean(ious)]
            else:
                shape_data[shape].append(np.mean(ious))
    
    du1, du2 = [], []
    for c in color_data:
        dm1, dm2 = np.mean(color_data[c]), np.var(color_data[c])
        du1.append(dm1)
        du2.append(dm2)
        print(dm1, dm2)
    print(np.mean(du1), np.mean(du2))
    print()
    du1, du2 = [], []
    for c in shape_data:
        dm1, dm2 = np.mean(shape_data[c]), np.var(shape_data[c])
        du1.append(dm1)
        du2.append(dm2)
        print(dm1, dm2)
    print(np.mean(du1), np.mean(du2))


def random_colors():
    colors = {}
    for i in range(7):
        r = random.randint(0,255)
        g = random.randint(0,255)
        b = random.randint(0,255)
        colors[torch.tensor([r, g, b])] = "c-%d" % i
    print(colors)

if __name__ == '__main__':
    analyze()