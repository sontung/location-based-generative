import torch
import numpy as np
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from models import LocationBasedGenerator
from data_loader import PBW, sim_collate_fn, SimData
from utils import show2, compute_iou

device = "cuda"
sae = LocationBasedGenerator()
sae.to(device)
sae.load_state_dict(torch.load("pre_models/model-sim-20200725-114336", map_location=device))
sae.eval()
# root_dir = "/home/sontung/Downloads"
root_dir = "/scratch/mlr/nguyensg/pbw"

data_dirs = [
    "%s/6objs_seg" % root_dir,
    "%s/7objs_7k" % root_dir,
    # "%s/6objs_view15degLeft" % root_dir,
    # "%s/6objs_view10degRight" % root_dir,
    # "%s/6objs_view5Left" % root_dir,
]


def eval_f(model_, iter_, nb_samples, device_="cuda"):
    total_loss = 0
    correct = 0.0

    count_2 = 0
    count_ = 0
    ious = []
    for idx, train_batch in enumerate(iter_):
        start, default, weight_maps = [tensor.to(device_) for tensor in train_batch[:3]]
        graphs, ob_names, im_names = train_batch[3:]
        with torch.no_grad():
            loss, start_pred = model_(start, default, weight_maps)
            pred_sg = model_.return_sg(start, ob_names)
        ious.extend(compute_iou(start_pred, start)[0])

        total_loss += loss.item()
        for i in range(len(graphs)):
            res = sorted(graphs[i]) == sorted(pred_sg[i])
            correct += res

    return total_loss, correct/nb_samples, np.mean(ious)


if __name__ == '__main__':
    for data_dir in data_dirs:
        val_data2 = SimData(train=False,
                            root_dir=data_dir,
                            train_size=0.0, nb_samples=20000, if_save_data=False)
        TOTAL = len(val_data2)
        val_iterator2 = DataLoader(val_data2, batch_size=16, shuffle=False, collate_fn=sim_collate_fn)
        res = eval_f(sae, val_iterator2, TOTAL, device)
        print(data_dir)
        print("  ", res, "\n")