from torch.utils.data import Dataset, DataLoader
from data_loader import collate_fn_trans, SimData, sim_collate_fn
from torchvision.utils import make_grid
from models import RearrangeNetwork, LocationBasedGenerator
from utils import compute_grad, show2
import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse
import datetime
from torch.utils.tensorboard import SummaryWriter


PARSER = argparse.ArgumentParser()
PARSER.add_argument("--dir", help="train directory",
                    default="/home/sontung/Downloads/5objs_seg", type=str)
PARSER.add_argument("--eval_dir", help="2nd domain evaluation directory",
                    default="/scratch/mlr/nguyensg/pbw/blocks-5-3", type=str)
PARSER.add_argument("--nb_samples", help="how many samples", default=10, type=int)
PARSER.add_argument("--epc", help="how many epochs", default=1, type=int)

PARSER.add_argument("--device", help="gpu device", default=0, type=int)

MY_ARGS = PARSER.parse_args()

NB_EPOCHS = MY_ARGS.epc
NB_SAMPLES = MY_ARGS.nb_samples
ROOT_DIR = MY_ARGS.dir
EVAL_DIR = MY_ARGS.eval_dir

# DEVICE = "cpu"
DEVICE = "cuda:%d" % MY_ARGS.device

print("Using", DEVICE)


def eval_f(model_, iter_, name_="1", device_="cuda", debugging=False):
    total_loss = 0
    vis = False
    correct = 0
    if not debugging:
        for idx, train_batch in enumerate(iter_):
            start, default, weight_maps = [tensor.to(device_) for tensor in train_batch[:3]]
            graphs, ob_names, _ = train_batch[3:]
            with torch.no_grad():
                loss, start_pred = model_(start, default, weight_maps)
                pred_sg = model_.return_sg(start, ob_names)
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
    else:
        scene_true = []
        scene_pred = []
        diff = []
        defaults = []
        weights = []
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
    return total_loss, correct


def train():
    now = datetime.datetime.now()
    writer = SummaryWriter("logs/sim" + now.strftime("%Y%m%d-%H%M%S") + "/")

    nb_epochs = NB_EPOCHS
    device = DEVICE

    train_data = SimData(root_dir=ROOT_DIR, nb_samples=NB_SAMPLES, train_size=0.95)
    train_iterator = DataLoader(train_data, batch_size=8, shuffle=True, collate_fn=sim_collate_fn)

    val_data2 = SimData(train=False, root_dir=ROOT_DIR, nb_samples=NB_SAMPLES, train_size=0.9)
    val_iterator2 = DataLoader(val_data2, batch_size=8, shuffle=True, collate_fn=sim_collate_fn)

    val_data = SimData(train=False, root_dir=EVAL_DIR, nb_samples=NB_SAMPLES, train_size=0.0)
    val_iterator = DataLoader(val_data, batch_size=16, shuffle=False, collate_fn=sim_collate_fn)

    model = LocationBasedGenerator()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    for epc in range(nb_epochs):

        model.train()
        total_loss = 0
        for idx, train_batch in enumerate(train_iterator):
            optimizer.zero_grad()
            start, default, weight_maps = [tensor.to(device) for tensor in train_batch[:3]]
            loss, start_pred = model(start, default, weight_maps)
            loss.backward()

            optimizer.step()

            total_loss += loss.item()
        writer.add_scalar('train/loss', total_loss/len(train_data), epc)

        model.eval()
        loss, acc = eval_f(model, val_iterator, name_=str(epc), device_=device, debugging=epc==nb_epochs-1)
        writer.add_scalar('val/loss', loss/len(val_data), epc)
        writer.add_scalar('val/acc', acc/len(val_data), epc)
        print(epc, acc/len(val_data), loss/len(val_data))

        loss, acc = eval_f(model, val_iterator2, name_=str(epc), device_=device)
        writer.add_scalar('val/loss', loss / len(val_data2), epc)
        writer.add_scalar('val/acc', acc / len(val_data2), epc)
        print(epc, acc / len(val_data2), loss / len(val_data2))

    torch.save(model.state_dict(), "pre_models/model-sim-%s" % now.strftime("%Y%m%d-%H%M%S"))


if __name__ == '__main__':
    train()
