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
from utils import show2, read_seg_masks


device = "cpu"
sae = LocationBasedGenerator()
sae.to(device)
sae.load_state_dict(torch.load("pre_models/model-sim-20200721-074631", map_location=device))

masks, def_mat, wei_mat, ob_names, sg_true = read_seg_masks()
sg = sae.return_sg(masks.unsqueeze(0), [ob_names])
print(sg)
print(sg_true)
