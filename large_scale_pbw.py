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
sae.load_state_dict(torch.load("pre_models/model-20200728-043622", map_location=device))
sae.eval()
# root_dir = "/home/sontung/Downloads"

data_dirs = [
    # "/home/sontung/thesis/photorealistic-blocksworld/blocks-6-3",
    "/scratch/mlr/nguyensg/pbw/blocks-6-3"
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
            if iou[i] > 0:
                insights[res].append(([idx, i], iou[i]))

    print(sorted(insights[0], key=lambda x: x[1], reverse=True)[:50])
    print(sorted(insights[1], key=lambda x: x[1], reverse=False)[:50])

    return total_loss, correct/nb_samples, np.mean(ious)

def analyze(model_, iter_, nb_samples, device_="cuda"):
    bad = [([19, 48], 0.8823529411764706), ([20, 21], 0.8823529411764706), ([51, 48], 0.8823529411764706), ([53, 57], 0.8823529411764706), ([67, 3], 0.8823529411764706), ([150, 3], 0.8823529411764706), ([158, 3], 0.8823529411764706), ([175, 57], 0.8823529411764706), ([222, 30], 0.8823529411764706), ([229, 48], 0.8823529411764706), ([232, 3], 0.8823529411764706), ([214, 29], 0.8814814814814815), ([105, 30], 0.87890625), ([121, 48], 0.87890625), ([127, 3], 0.87890625), ([195, 12], 0.87890625), ([240, 3], 0.87890625), ([253, 12], 0.87890625), ([298, 30], 0.87890625), ([10, 3], 0.8784313725490196), ([33, 21], 0.8784313725490196), ([42, 30], 0.8784313725490196), ([75, 48], 0.8784313725490196), ([108, 3], 0.8784313725490196), ([204, 30], 0.8784313725490196), ([210, 21], 0.8784313725490196), ([211, 48], 0.8784313725490196), ([5, 30], 0.875), ([17, 57], 0.875), ([40, 3], 0.875), ([40, 21], 0.875), ([40, 48], 0.875), ([52, 3], 0.875), ([57, 39], 0.875), ([59, 48], 0.875), ([60, 57], 0.875), ([61, 3], 0.875), ([65, 12], 0.875), ([80, 57], 0.875), ([81, 30], 0.875), ([81, 57], 0.875), ([104, 12], 0.875), ([104, 48], 0.875), ([105, 57], 0.875), ([113, 30], 0.875), ([121, 57], 0.875), ([142, 48], 0.875), ([147, 57], 0.875), ([164, 48], 0.875), ([165, 12], 0.875)]
    good = [([4, 32], 0.04245283018867924), ([10, 32], 0.04245283018867924), ([29, 51], 0.04245283018867924), ([34, 4], 0.04245283018867924), ([67, 32], 0.04245283018867924), ([77, 6], 0.04245283018867924), ([83, 7], 0.04245283018867924), ([148, 23], 0.04245283018867924), ([156, 32], 0.04245283018867924), ([159, 61], 0.04245283018867924), ([164, 5], 0.04245283018867924), ([203, 33], 0.04245283018867924), ([229, 43], 0.04245283018867924), ([233, 5], 0.04245283018867924), ([263, 13], 0.04245283018867924), ([280, 61], 0.04245283018867924), ([285, 60], 0.04245283018867924), ([64, 59], 0.04310344827586207), ([191, 6], 0.04310344827586207), ([86, 5], 0.04330708661417323), ([96, 52], 0.04330708661417323), ([146, 59], 0.04330708661417323), ([195, 15], 0.04330708661417323), ([259, 60], 0.04330708661417323), ([267, 41], 0.04330708661417323), ([62, 26], 0.06666666666666667), ([71, 53], 0.07692307692307693), ([52, 34], 0.08035714285714286), ([102, 59], 0.08035714285714286), ([153, 15], 0.08035714285714286), ([161, 43], 0.08035714285714286), ([165, 7], 0.08035714285714286), ([264, 23], 0.08035714285714286), ([286, 42], 0.08035714285714286), ([55, 14], 0.09009009009009009), ([249, 14], 0.09009009009009009), ([238, 42], 0.09053497942386832), ([175, 42], 0.0912863070539419), ([278, 40], 0.0989010989010989), ([194, 31], 0.09950248756218906), ([35, 61], 0.1), ([50, 61], 0.1), ([60, 23], 0.1), ([168, 60], 0.1), ([185, 16], 0.1), ([240, 5], 0.1), ([14, 48], 0.10294117647058823), ([57, 12], 0.10294117647058823), ([140, 3], 0.13176470588235295), ([225, 3], 0.13176470588235295)]

    bad_idx = [dm[0] for dm in bad]
    good_idx = [dm[0] for dm in good]

    total_loss = 0
    correct = 0.0

    ious = []

    for idx, train_batch in enumerate(iter_):
        start, coord_true, default, weight_maps = [tensor.to(device_) for tensor in train_batch[:4]]
        graphs, ob_names = train_batch[4:]
        with torch.no_grad():
            loss, start_pred = model_(start, default, weight_maps)
            pred_sg = model_.return_sg(start, ob_names)
        for i in range(len(graphs)):

            if [idx, i] in good_idx:
                print("good", [idx,i], pred_sg[i])
                show2([start[i].cpu(),
                       start_pred[i].cpu(),
                       start[i].cpu()+start_pred[i].cpu()], "figures_corr/%d-%d.png" % (idx, i), 1)
            elif [idx, i] in bad_idx:
                print([idx,i], pred_sg[i], graphs[i])
                show2([start[i].cpu(),
                       start_pred[i].cpu(),
                       start[i].cpu()+start_pred[i].cpu()], "figures/%d-%d.png" % (idx, i), 1)


if __name__ == '__main__':
    for data_dir in data_dirs:
        val_data2 = PBW(train=False,
                        root_dir=data_dir,
                        train_size=0.0, nb_samples=20160, if_save_data=False)
        TOTAL = len(val_data2)
        val_iterator2 = DataLoader(val_data2, batch_size=64, shuffle=False, collate_fn=pbw_collate_fn)
        analyze(sae, val_iterator2, TOTAL, device)
        # print(res)
