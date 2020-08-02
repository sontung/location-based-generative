
import torch
from torch.utils.data import Dataset, DataLoader
from models import LocationBasedGenerator
from data_loader import PBW, pbw_collate_fn, SimData, sim_collate_fn
from utils import show2

device = "cuda"
sae = LocationBasedGenerator()
sae.to(device)
sae.load_state_dict(torch.load("pre_models/model-sim-20200725-114336", map_location=device))
sae.eval()
val_data2 = SimData(train=False, root_dir="/home/sontung/Downloads/6objs_seg",
                train_size=0.0, nb_samples=10000)
val_iterator2 = DataLoader(val_data2, batch_size=16, shuffle=False, collate_fn=sim_collate_fn)

def eval(model_, iter_, device_="cuda"):
    correct = 0
    count_ = 0
    for idx, train_batch in enumerate(iter_):
        start, default, weight_maps = [tensor.to(device_) for tensor in train_batch[:3]]
        graphs, ob_names, im_names = train_batch[3:]

        with torch.no_grad():
            loss, start_pred = model_(start, default, weight_maps)
            pred_sg = model_.return_sg(start, ob_names)

        for i in range(len(graphs)):
            res = sorted(graphs[i]) == sorted(pred_sg[i])
            correct += res
            if res == 0:
                count_ += 1
                if count_ <= 50:
                    print(sorted(graphs[i]))
                    print(sorted(pred_sg[i]))
                    print()
                    show2([
                        torch.sum(start[i], dim=0).unsqueeze(0).cpu(),
                        torch.sum(start_pred[i], dim=0).unsqueeze(0).cpu(),
                        start[i].cpu(),
                        start_pred[i].cpu(),
                        start[i].cpu() + start_pred[i].cpu(),
                        default[i].cpu(),
                        weight_maps[i].cpu()
                    ], "figures/debug-%d.png" % count_, 4)
    print(correct)
    return correct

eval(sae, val_iterator2, device)