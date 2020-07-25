import torch

from torch.utils.data import Dataset, DataLoader
from models import LocationBasedGenerator
from data_loader import PBW, sim_collate_fn, SimData
from utils import show2

device = "cuda"
sae = LocationBasedGenerator()
sae.to(device)
sae.load_state_dict(torch.load("pre_models/model-sim-20200721-164922", map_location=device))

val_data2 = SimData(train=False,
                    root_dir="/home/sontung/Downloads/7objs_7k",
                    train_size=0.0, nb_samples=1000)
TOTAL = len(val_data2)
val_iterator2 = DataLoader(val_data2, batch_size=16, shuffle=False, collate_fn=sim_collate_fn)

def eval_f(model_, iter_, device_="cuda"):
    total_loss = 0
    correct = 0.0

    count_2 = 0
    count_ = 0
    for idx, train_batch in enumerate(iter_):
        start, default, weight_maps = [tensor.to(device_) for tensor in train_batch[:3]]
        graphs, ob_names, im_names = train_batch[3:]
        with torch.no_grad():
            loss, start_pred = model_(start, default, weight_maps)
            pred_sg = model_.return_sg(start, ob_names)
        total_loss += loss.item()
        for i in range(len(graphs)):
            res = sorted(graphs[i]) == sorted(pred_sg[i])
            correct += res
            if res == 0:
                count_ += 1
                if count_ <= 50:
                    print(count_, im_names[i], sorted(graphs[i]), sorted(pred_sg[i]), "\n")
                    show2([
                        torch.sum(start[i], dim=0).unsqueeze(0).cpu(),
                        torch.sum(start_pred[i], dim=0).unsqueeze(0).cpu(),
                        start[i].cpu(),
                        start_pred[i].cpu(),
                        start[i].cpu() + start_pred[i].cpu(),
                        default[i].cpu(),
                        weight_maps[i].cpu()
                    ], "figures/debug-%d.png" % count_, 4)
            else:
                count_2 += 1
                if count_2 <= 50:
                    show2([
                        torch.sum(start[i], dim=0).unsqueeze(0).cpu(),
                        torch.sum(start_pred[i], dim=0).unsqueeze(0).cpu(),
                        start[i].cpu(),
                        start_pred[i].cpu(),
                        start[i].cpu() + start_pred[i].cpu(),
                        default[i].cpu(),
                        weight_maps[i].cpu()
                    ], "figures_corr/debug-%d.png" % count_2, 4)

    print(correct/TOTAL)
    return total_loss, correct

eval_f(sae, val_iterator2, device)