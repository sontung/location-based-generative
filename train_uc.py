from torch.utils.data import Dataset, DataLoader
from data_loader import collate_fn_trans, PBW, pbw_collate_fn
from torchvision.utils import make_grid
from models import LocationBasedGeneratorCoordConv, LocationBasedGenerator
from utils import compute_grad, show2, compute_iou
import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse
import datetime
from torch.utils.tensorboard import SummaryWriter


PARSER = argparse.ArgumentParser()
PARSER.add_argument("--dir", help="train directory",
                    default="/home/sontung/thesis/photorealistic-blocksworld/blocks-5-3", type=str)
PARSER.add_argument("--eval_dir", help="2nd domain evaluation directory",
                    default="/home/sontung/thesis/photorealistic-blocksworld/blocks-6-3-old", type=str)
PARSER.add_argument("--nb_samples", help="how many samples", default=3000, type=int)
PARSER.add_argument("--device", help="gpu device", default=0, type=int)
PARSER.add_argument("--save_data", help="wheather to save processed data", default=1, type=int)

MY_ARGS = PARSER.parse_args()

NB_SAMPLES = MY_ARGS.nb_samples
ROOT_DIR = MY_ARGS.dir
EVAL_DIR = MY_ARGS.eval_dir
DEVICE = "cuda:%d" % MY_ARGS.device
SAVE_DATA = MY_ARGS.save_data == 1


def eval_f(model_, iter_, name_="1", device_="cuda"):
    total_loss = 0
    vis = False
    correct = 0
    ious = []
    for idx, train_batch in enumerate(iter_):
        start, coord_true, default, weight_maps = [tensor.to(device_) for tensor in train_batch[:4]]
        graphs, ob_names = train_batch[4:]
        with torch.no_grad():
            loss, start_pred = model_(start, default, weight_maps)
            pred_sg = model_.return_sg(start, ob_names)
        ious.extend(compute_iou(start_pred, start)[0])
        total_loss += loss.item()
        for i in range(len(graphs)):
            correct += sorted(graphs[i]) == sorted(pred_sg[i])

        if not vis:
            show2([
                torch.sum(start, dim=1)[:16].cpu(),
                torch.sum(start_pred, dim=1)[:16].detach().cpu(),
                start_pred.detach().cpu().view(-1, 3, 128, 128)[:16]+start.cpu().view(-1, 3, 128, 128)[:16],
                default.cpu().view(-1, 3, 128, 128)[:16],
                weight_maps.cpu().view(-1, 1, 128, 128)[:16]
            ], "figures/test%s.png" % name_, 4)
            vis = True
    return total_loss, correct, np.mean(ious)


def train():
    now = datetime.datetime.now()
    writer = SummaryWriter("logs/" + now.strftime("%Y%m%d-%H%M%S") + "/")

    nb_epochs = 20
    device = DEVICE

    train_data = PBW(root_dir=ROOT_DIR, nb_samples=NB_SAMPLES, train_size=1.0)
    train_iterator = DataLoader(train_data, batch_size=8, shuffle=True, collate_fn=pbw_collate_fn)

    val_data2 = PBW(train=False, root_dir=EVAL_DIR, train_size=0.0, nb_samples=200, if_save_data=SAVE_DATA)
    val_iterator2 = DataLoader(val_data2, batch_size=16, shuffle=False, collate_fn=pbw_collate_fn)
    model = LocationBasedGenerator()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    for epc in range(nb_epochs):

        model.train()
        for idx, train_batch in enumerate(train_iterator):
            optimizer.zero_grad()
            start, coord_true, default, weight_maps = [tensor.to(device) for tensor in train_batch[:4]]

            loss, start_pred = model(start, default, weight_maps)
            print(start.size())
            loss.backward()

            optimizer.step()
            iou = compute_iou(start_pred, start)[1]
            writer.add_scalar('train/loss', loss.item(), idx + epc * len(train_iterator))
            writer.add_scalar('train/iou', iou, idx + epc * len(train_iterator))

        model.eval()

        loss, acc, iou = eval_f(model, val_iterator2, name_="-d-"+str(epc), device_=device)
        writer.add_scalar('val/loss', loss / len(val_data2), epc)
        writer.add_scalar('val/acc', acc / len(val_data2), epc)
        writer.add_scalar('val/iou', iou, epc)

        print(epc, acc / len(val_data2), iou)

    torch.save(model.state_dict(), "pre_models/model-%s" % now.strftime("%Y%m%d-%H%M%S"))



if __name__ == '__main__':
    train()