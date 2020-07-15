from torch.utils.data import Dataset, DataLoader
from data_loader import PBWtransition, collate_fn_trans
from torchvision.utils import make_grid
from models import RearrangeNetwork

import matplotlib.pyplot as plt
import numpy as np
import torch

def show2(im_, name, nrow):
    import logging

    logger = logging.getLogger()
    old_level = logger.level
    logger.setLevel(100)

    fig_ = plt.figure(figsize=(15, 15))
    for du3 in range(1, len(im_)+1):
        plt.subplot(1, len(im_), du3)
        plt.axis("off")
        plt.imshow(np.transpose(make_grid(im_[du3-1], padding=5, normalize=False, pad_value=50, nrow=nrow),
                                (1, 2, 0)))

    plt.axis("off")
    # plt.title("black: no action, red: 1-3, yellow: 3-1, green: 1-2, blue: 2-3, pink: 3-2, brown: 2-1")
    plt.savefig(name, transparent=True, bbox_inches='tight')
    plt.close(fig_)
    logger.setLevel(old_level)

def train():
    nb_epochs = 1
    device = "cuda"

    train = PBWtransition()
    train_iterator = DataLoader(train, batch_size=64, shuffle=True, collate_fn=collate_fn_trans)

    model = RearrangeNetwork()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)

    for epc in range(nb_epochs):

        model.train()
        for idx, train_batch in enumerate(train_iterator):
            optimizer.zero_grad()
            start, end, act = [tensor.to(device) for tensor in train_batch]
            end_sum = torch.sum(end, dim=1)

            loss, end_pred = model(start, act, end_sum)
            loss.backward()
            optimizer.step()

            print(loss.item())
            show2([torch.sum(start, dim=1)[:16].cpu(),
                   end_sum[:16].cpu(),
                   end_pred[:16].detach().cpu()], "test.png", 4)


if __name__ == '__main__':
    train()