from torch.utils.data import Dataset, DataLoader
from data_loader import PBWtransition, collate_fn_trans, PBW, pbw_collate_fn, PBW_AmbMasked
from torchvision.utils import make_grid
from models import RearrangeNetwork, LocationBasedGenerator, Transformer, FindTopNetwork
from utils import compute_grad, show2
import matplotlib.pyplot as plt
import numpy as np
import torch

def eval_ad(net_, data_iter, device="cuda"):
    acc_list = []
    loss_list = []
    total = 0.0
    loss_fn_ = torch.nn.BCEWithLogitsLoss()

    for idx, train_batch in enumerate(data_iter):
        masks, targets, _ = [tensor.to(device) for tensor in train_batch]
        with torch.no_grad():
            pred = net_(masks)
            loss = loss_fn_(pred, targets)
            acc = torch.sum(torch.all(torch.eq(torch.sigmoid(pred) > 0.5, targets))).item()
        loss_list.append(loss.item())
        acc_list.append(acc)
        total += masks.size(0)
    print(np.sum(acc_list)/total)
    return np.sum(acc_list) / total / 6, np.mean(loss_list)

def train():
    nb_epochs = 100
    device = "cuda"

    train = PBW_AmbMasked()
    train_iterator = DataLoader(train, batch_size=64, shuffle=True, collate_fn=pbw_collate_fn)
    val_iterator = DataLoader(PBW_AmbMasked(train=False), batch_size=64, shuffle=True, collate_fn=pbw_collate_fn)


    model = FindTopNetwork()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1, weight_decay=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for epc in range(nb_epochs):

        model.train()
        total_loss = 0
        for idx, train_batch in enumerate(train_iterator):
            optimizer.zero_grad()
            masks, targets, imgs = [tensor.to(device) for tensor in train_batch]
            targets = targets.float()
            preds = model(masks)
            loss = loss_fn(preds, targets)
            loss.backward()

            optimizer.step()

            total_loss += loss.item()
        print(total_loss)
        model.eval()
        eval_ad(model, val_iterator)



if __name__ == '__main__':
    train()