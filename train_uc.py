from torch.utils.data import Dataset, DataLoader
from data_loader import PBWtransition, collate_fn_trans, PBW, pbw_collate_fn
from torchvision.utils import make_grid
from models import RearrangeNetwork, LocationBasedGenerator, Transformer
from utils import compute_grad, show2
import matplotlib.pyplot as plt
import numpy as np
import torch



def eval(model_, iter_, name_=1, device_="cuda"):
    total_loss = 0
    vis = False
    for idx, train_batch in enumerate(iter_):
        start, coord_true, default, weight_maps = [tensor.to(device_) for tensor in train_batch]
        loss, start_pred = model_(start, default, weight_maps)
        total_loss += loss.item()
        if not vis:
            show2([
                torch.sum(start, dim=1)[:16].cpu(),
                torch.sum(start_pred, dim=1)[:16].detach().cpu(),
                start_pred.detach().cpu().view(-1, 3, 128, 128)[:16]+start.cpu().view(-1, 3, 128, 128)[:16],
            ], "figures/test%d.png" % name_, 4)
            vis = True
    return total_loss

def train():
    nb_epochs = 1000
    device = "cuda"

    train_data = PBW()
    train_iterator = DataLoader(train_data, batch_size=8, shuffle=True, collate_fn=pbw_collate_fn)
    val_data = PBW(train=False)
    val_iterator = DataLoader(val_data, batch_size=16, shuffle=False, collate_fn=pbw_collate_fn)
    model = LocationBasedGenerator()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    for epc in range(nb_epochs):

        model.train()
        total_loss = 0
        for idx, train_batch in enumerate(train_iterator):
            optimizer.zero_grad()
            start, coord_true, default, weight_maps = [tensor.to(device) for tensor in train_batch]

            loss, start_pred = model(start, default, weight_maps)
            loss.backward()

            optimizer.step()

            total_loss += loss.item()
        # print(total_loss/len(train_data))

        model.eval()
        loss = eval(model, val_iterator, name_=epc)
        print("val", loss/len(val_data))



if __name__ == '__main__':
    train()