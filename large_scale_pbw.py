import torch
import numpy as np
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from models import LocationBasedGenerator
from data_loader import PBW, pbw_collate_fn, SimData
from utils import show2, compute_iou

device = "cuda"
sae = LocationBasedGenerator()
sae.to(device)
sae.load_state_dict(torch.load("pre_models/model-sim-20200725-114336", map_location=device))

# root_dir = "/home/sontung/Downloads"

data_dirs = [
    # "/home/sontung/thesis/photorealistic-blocksworld/blocks-6-3",
    "/scratch/mlr/nguyensg/data/blocks-6-3"
]


def eval_f(model_, iter_, nb_samples, device_="cuda"):
    total_loss = 0
    correct = 0.0

    count_2 = 0
    count_ = 0
    ious = []
    insights =[[], []]
    for idx, train_batch in enumerate(iter_):
        start, coord_true, default, weight_maps = [tensor.to(device_) for tensor in train_batch[:4]]
        graphs, ob_names = train_batch[4:]
        with torch.no_grad():
            loss, start_pred = model_(start, default, weight_maps)
            pred_sg = model_.return_sg(start, ob_names)
        iou = compute_iou(start_pred, start)[0]
        ious.extend(iou)

        total_loss += loss.item()
        for i in range(len(graphs)):
            res = sorted(graphs[i]) == sorted(pred_sg[i])
            correct += res
            insights[res].append(([idx, i], iou[i]))

    print(sorted(insights[0], key=lambda x: x[1])[:20])
    print(sorted(insights[1], key=lambda x: x[1], reverse=True)[:20])

    return total_loss, correct/nb_samples, np.mean(ious)


if __name__ == '__main__':
    for data_dir in data_dirs:
        val_data2 = PBW(train=False,
                        root_dir=data_dir,
                        train_size=0.0, nb_samples=20160, if_save_data=False)
        TOTAL = len(val_data2)
        val_iterator2 = DataLoader(val_data2, batch_size=64, shuffle=False, collate_fn=pbw_collate_fn)
        res = eval_f(sae, val_iterator2, TOTAL, device)
        print(res)