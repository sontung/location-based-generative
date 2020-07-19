import torch

from torch.utils.data import Dataset, DataLoader
from models import LocationBasedGenerator
from data_loader import PBW, pbw_collate_fn

device = "cuda"
sae = LocationBasedGenerator()
sae.to(device)
sae.load_state_dict(torch.load("pre_models/model-20200718-122629", map_location=device))

val_data2 = PBW(train=False, root_dir="/home/sontung/thesis/photorealistic-blocksworld/blocks-5-3",
                train_size=0.0, nb_samples=1000)
val_iterator2 = DataLoader(val_data2, batch_size=16, shuffle=False, collate_fn=pbw_collate_fn)

def eval(model_, iter_, device_="cuda"):
    correct = 0
    for idx, train_batch in enumerate(iter_):
        start, coord_true, default, weight_maps = [tensor.to(device_) for tensor in train_batch[:4]]
        graphs, ob_names = train_batch[4:]
        with torch.no_grad():
            pred_sg = model_.return_sg(start, ob_names)
            print(start.size(), len(ob_names[0]))
            print(ob_names)
        for i in range(len(graphs)):
            correct += sorted(graphs[i]) == sorted(pred_sg[i])
    print(correct)
    return correct

eval(sae, val_iterator2, device)