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
sae.load_state_dict(torch.load("pre_models/model-20200718-122629", map_location=device))

masks, names, sg_true = read_seg_masks()
sg = sae.return_sg(masks.unsqueeze(0), [names])
print(sg)
print(sg_true)
