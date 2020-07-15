from torch.utils.data import Dataset, DataLoader
from data_loader import PBWtransition, collate_fn_trans, PBW, pbw_collate_fn
from torchvision.utils import make_grid
from models import RearrangeNetwork, LocationBasedGenerator, Transformer
from utils import compute_grad
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
    nb_epochs = 100
    device = "cuda"

    train = PBW()
    train_iterator = DataLoader(train, batch_size=8, shuffle=True, collate_fn=pbw_collate_fn)

    model = LocationBasedGenerator()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1, weight_decay=1e-4)

    for epc in range(nb_epochs):

        model.train()
        total_loss = 0
        for idx, train_batch in enumerate(train_iterator):
            optimizer.zero_grad()
            start, coord_true, default = [tensor.to(device) for tensor in train_batch]

            loss, start_pred = model(start, default)
            loss.backward()

            optimizer.step()

            total_loss += loss.item()
        print(total_loss)
        show2([
            start[:16, :3, :].cpu(),
            default[:16, :3, :].cpu(),
            start_pred[:16].detach().cpu()
        ], "test.png", 4)


if __name__ == '__main__':
    train()