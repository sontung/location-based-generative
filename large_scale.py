import torch
import numpy as np
import gc
import pickle
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from models import LocationBasedGenerator
from data_loader import PBW, sim_collate_fn, SimData, SimDataEval, SimDataEvalDebug
from utils import show2, compute_iou

device = "cuda"
sae = LocationBasedGenerator()
sae.to(device)
sae.load_state_dict(torch.load("pre_models/model-sim-20200725-114336", map_location=device))
sae.eval()
root_dir = "/home/sontung/Downloads"
# root_dir = "/scratch/mlr/nguyensg/pbw"

data_dirs = [
    "%s/6objs_seg" % root_dir,
    # "%s/7objs_7k" % root_dir,
    # "%s/6objs_view15degLeft" % root_dir,
    # "%s/6objs_view10degRight" % root_dir,
    # "%s/6objs_view5Left" % root_dir,
]


def eval_f(model_, iter_, nb_samples, device_="cuda"):
    total_loss = 0
    correct = 0.0

    count_2 = 0
    count_ = 0
    ious = []
    ious2 = []
    wrong = []
    for idx, train_batch in enumerate(iter_):
        start, default, weight_maps = [tensor.to(device_) for tensor in train_batch[:3]]
        graphs, ob_names, im_names = train_batch[3:]
        with torch.no_grad():
            loss, start_pred = model_(start, default, weight_maps)
            pred_sg = model_.return_sg(start, ob_names)
        iou = compute_iou(start_pred, start)[0]

        total_loss += loss.item()
        for i in range(len(graphs)):
            res = sorted(graphs[i]) == sorted(pred_sg[i])
            correct += res
            if res == 0:
                wrong.append(im_names[i])
                ious2.append(iou[i])
            elif res == 1:
                ious.append(iou[i])

    return ious, ious2, wrong


def analyze(model_):
    wrong_list = ['z.seg385_s0_310254_s1__s2_.ppm', 'z.seg2049_s2_502341_s0__s1_.ppm', 'z.seg289_s0_230154_s1__s2_.ppm',
     'z.seg1069_s1_253041_s2__s0_.ppm', 'z.seg349_s0_253041_s1__s2_.ppm', 'z.seg255_s0_204351_s1__s2_.ppm',
     'z.seg367_s0_302154_s1__s2_.ppm', 'z.seg615_s0_503241_s1__s2_.ppm', 'z.seg1371_s1_520341_s2__s0_.ppm',
     'z.seg121_s0_102354_s1__s2_.ppm', 'z.seg1691_s2_203541_s0__s1_.ppm', 'z.seg1867_s2_325041_s0__s1_.ppm',
     'z.seg1551_s2_053241_s0__s1_.ppm', 'z.seg45_s0_025341_s1__s2_.ppm', 'z.seg445_s0_342051_s1__s2_.ppm',
     'z.seg1590_s2_123045_s0__s1_.ppm', 'z.seg261_s0_205341_s1__s2_.ppm', 'z.seg1909_s2_352041_s0__s1_.ppm',
     'z.seg1104_s1_310245_s2__s0_.ppm', 'z.seg990_s1_213045_s2__s0_.ppm', 'z.seg1830_s2_312045_s0__s1_.ppm',
     'z.seg381_s0_305241_s1__s2_.ppm', 'z.seg1899_s2_350241_s0__s1_.ppm', 'z.seg1110_s1_312045_s2__s0_.ppm',
     'z.seg960_s1_201345_s2__s0_.ppm', 'z.seg1091_s1_302541_s2__s0_.ppm', 'z.seg411_s0_320451_s1__s2_.ppm',
     'z.seg750_s1_023145_s2__s0_.ppm', 'z.seg427_s0_325041_s1__s2_.ppm', 'z.seg271_s0_213054_s1__s2_.ppm',
     'z.seg39_s0_024351_s1__s2_.ppm', 'z.seg375_s0_304251_s1__s2_.ppm', 'z.seg293_s0_230541_s1__s2_.ppm',
     'z.seg1446_s2_013245_s0__s1_.ppm', 'z.seg1853_s2_320541_s0__s1_.ppm', 'z.seg1811_s2_302541_s0__s1_.ppm',
     'z.seg1854_s2_321045_s0__s1_.ppm', 'z.seg87_s0_043251_s1__s2_.ppm', 'z.seg1614_s2_132045_s0__s1_.ppm',
     'z.seg489_s0_402351_s1__s2_.ppm', 'z.seg459_s0_350241_s1__s2_.ppm', 'z.seg57_s0_032451_s1__s2_.ppm',
     'z.seg145_s0_120354_s1__s2_.ppm', 'z.seg175_s0_132054_s1__s2_.ppm', 'z.seg755_s1_023541_s2__s0_.ppm',
     'z.seg1133_s1_320541_s2__s0_.ppm', 'z.seg1734_s2_231045_s0__s1_.ppm', 'z.seg1494_s2_032145_s0__s1_.ppm',
     'z.seg888_s1_130245_s2__s0_.ppm', 'z.seg1728_s2_230145_s0__s1_.ppm', 'z.seg295_s0_231054_s1__s2_.ppm',
     'z.seg1821_s2_305241_s0__s1_.ppm', 'z.seg1013_s1_230541_s2__s0_.ppm', 'z.seg768_s1_031245_s2__s0_.ppm',
     'z.seg720_s1_012345_s2__s0_.ppm', 'z.seg1686_s2_203145_s0__s1_.ppm', 'z.seg1101_s1_305241_s2__s0_.ppm',
     'z.seg1086_s1_302145_s2__s0_.ppm', 'z.seg541_s0_423051_s1__s2_.ppm', 'z.seg864_s1_120345_s2__s0_.ppm',
     'z.seg1584_s2_120345_s0__s1_.ppm', 'z.seg1329_s1_502341_s2__s0_.ppm', 'z.seg63_s0_034251_s1__s2_.ppm',
     'z.seg413_s0_320541_s1__s2_.ppm', 'z.seg49_s0_031254_s1__s2_.ppm', 'z.seg1806_s2_302145_s0__s1_.ppm',
     'z.seg421_s0_324051_s1__s2_.ppm', 'z.seg1509_s2_035241_s0__s1_.ppm', 'z.seg1848_s2_320145_s0__s1_.ppm',
     'z.seg435_s0_340251_s1__s2_.ppm', 'z.seg369_s0_302451_s1__s2_.ppm', 'z.seg661_s0_523041_s1__s2_.ppm',
     'z.seg779_s1_032541_s2__s0_.ppm', 'z.seg409_s0_320154_s1__s2_.ppm', 'z.seg555_s0_430251_s1__s2_.ppm',
     'z.seg265_s0_210354_s1__s2_.ppm', 'z.seg371_s0_302541_s1__s2_.ppm', 'z.seg685_s0_532041_s1__s2_.ppm',
     'z.seg7_s0_013254_s1__s2_.ppm', 'z.seg495_s0_403251_s1__s2_.ppm', 'z.seg846_s1_103245_s2__s0_.ppm',
     'z.seg1747_s2_235041_s0__s1_.ppm', 'z.seg1701_s2_205341_s0__s1_.ppm', 'z.seg241_s0_201354_s1__s2_.ppm',
     'z.seg1395_s1_530241_s2__s0_.ppm', 'z.seg1499_s2_032541_s0__s1_.ppm', 'z.seg1704_s2_210345_s0__s1_.ppm',
     'z.seg169_s0_130254_s1__s2_.ppm', 'z.seg1566_s2_103245_s0__s1_.ppm', 'z.seg2101_s2_523041_s0__s1_.ppm',
     'z.seg825_s1_052341_s2__s0_.ppm', 'z.seg247_s0_203154_s1__s2_.ppm', 'z.seg1608_s2_130245_s0__s1_.ppm',
     'z.seg69_s0_035241_s1__s2_.ppm', 'z.seg315_s0_240351_s1__s2_.ppm', 'z.seg609_s0_502341_s1__s2_.ppm',
     'z.seg301_s0_234051_s1__s2_.ppm', 'z.seg1789_s2_253041_s0__s1_.ppm', 'z.seg25_s0_021354_s1__s2_.ppm',
     'z.seg984_s1_210345_s2__s0_.ppm', 'z.seg981_s1_205341_s2__s0_.ppm', 'z.seg1335_s1_503241_s2__s0_.ppm',
     'z.seg31_s0_023154_s1__s2_.ppm', 'z.seg1440_s2_012345_s0__s1_.ppm', 'z.seg1710_s2_213045_s0__s1_.ppm',
     'z.seg33_s0_023451_s1__s2_.ppm', 'z.seg1014_s1_231045_s2__s0_.ppm', 'z.seg971_s1_203541_s2__s0_.ppm',
     'z.seg81_s0_042351_s1__s2_.ppm', 'z.seg291_s0_230451_s1__s2_.ppm', 'z.seg1059_s1_250341_s2__s0_.ppm',
     'z.seg391_s0_312054_s1__s2_.ppm', 'z.seg870_s1_123045_s2__s0_.ppm', 'z.seg1381_s1_523041_s2__s0_.ppm',
     'z.seg1733_s2_230541_s0__s1_.ppm', 'z.seg105_s0_052341_s1__s2_.ppm', 'z.seg2115_s2_530241_s0__s1_.ppm',
     'z.seg1488_s2_031245_s0__s1_.ppm', 'z.seg325_s0_243051_s1__s2_.ppm', 'z.seg1824_s2_310245_s0__s1_.ppm',
     'z.seg1800_s2_301245_s0__s1_.ppm', 'z.seg831_s1_053241_s2__s0_.ppm', 'z.seg2091_s2_520341_s0__s1_.ppm',
     'z.seg1_s0_012354_s1__s2_.ppm', 'z.seg1485_s2_025341_s0__s1_.ppm', 'z.seg1008_s1_230145_s2__s0_.ppm',
     'z.seg35_s0_023541_s1__s2_.ppm', 'z.seg966_s1_203145_s2__s0_.ppm', 'z.seg1560_s2_102345_s0__s1_.ppm',
     'z.seg2125_s2_532041_s0__s1_.ppm', 'z.seg2055_s2_503241_s0__s1_.ppm', 'z.seg1405_s1_532041_s2__s0_.ppm',
     'z.seg531_s0_420351_s1__s2_.ppm', 'z.seg1680_s2_201345_s0__s1_.ppm', 'z.seg127_s0_103254_s1__s2_.ppm',
     'z.seg339_s0_250341_s1__s2_.ppm', 'z.seg726_s1_013245_s2__s0_.ppm', 'z.seg1080_s1_301245_s2__s0_.ppm',
     'z.seg1470_s2_023145_s0__s1_.ppm', 'z.seg1027_s1_235041_s2__s0_.ppm', 'z.seg59_s0_032541_s1__s2_.ppm',
     'z.seg774_s1_032145_s2__s0_.ppm', 'z.seg151_s0_123054_s1__s2_.ppm', 'z.seg1189_s1_352041_s2__s0_.ppm',
     'z.seg840_s1_102345_s2__s0_.ppm', 'z.seg111_s0_053241_s1__s2_.ppm', 'z.seg1545_s2_052341_s0__s1_.ppm',
     'z.seg1779_s2_250341_s0__s1_.ppm', 'z.seg307_s0_235041_s1__s2_.ppm', 'z.seg651_s0_520341_s1__s2_.ppm',
     'z.seg894_s1_132045_s2__s0_.ppm', 'z.seg565_s0_432051_s1__s2_.ppm', 'z.seg1147_s1_325041_s2__s0_.ppm',
     'z.seg249_s0_203451_s1__s2_.ppm', 'z.seg415_s0_321054_s1__s2_.ppm', 'z.seg1128_s1_320145_s2__s0_.ppm',
     'z.seg55_s0_032154_s1__s2_.ppm', 'z.seg1464_s2_021345_s0__s1_.ppm', 'z.seg1475_s2_023541_s0__s1_.ppm',
     'z.seg469_s0_352041_s1__s2_.ppm', 'z.seg1179_s1_350241_s2__s0_.ppm', 'z.seg765_s1_025341_s2__s0_.ppm',
     'z.seg361_s0_301254_s1__s2_.ppm', 'z.seg1134_s1_321045_s2__s0_.ppm', 'z.seg744_s1_021345_s2__s0_.ppm',
     'z.seg789_s1_035241_s2__s0_.ppm', 'z.seg251_s0_203541_s1__s2_.ppm', 'z.seg675_s0_530241_s1__s2_.ppm']
    device_ = "cuda"
    val_data2 = SimDataEvalDebug(data_dirs[0], wrong_list)
    val_iterator2 = DataLoader(val_data2, batch_size=64, shuffle=False, collate_fn=sim_collate_fn)
    count_ = 0
    for idx, train_batch in enumerate(val_iterator2):
        start, default, weight_maps = [tensor.to(device_) for tensor in train_batch[:3]]
        graphs, ob_names, im_names = train_batch[3:]
        with torch.no_grad():
            loss, start_pred = model_(start, default, weight_maps)
            pred_sg = model_.return_sg(start, ob_names)
        iou = compute_iou(start_pred, start)[0]
        for i in range(len(graphs)):
            res = sorted(graphs[i]) == sorted(pred_sg[i])
            count_ += 1
            show2([
                torch.sum(start[i], dim=0).unsqueeze(0).cpu(),
                torch.sum(start_pred[i], dim=0).unsqueeze(0).cpu(),
                start[i].cpu(),
                start_pred[i].cpu(),
                start[i].cpu() + start_pred[i].cpu(),
                default[i].cpu(),
                weight_maps[i].cpu()
            ], "figures/debug-%d.png" % count_, 4)


if __name__ == '__main__':
    # analyze(sae)
    for data_dir in data_dirs:
        step = 2000
        max_step = 20000

        all_wrong = []
        corr_ious = []
        wrong_ious = []
        for du33 in range(0, max_step, step):

            val_data2 = SimDataEval(train=False, start_idx=du33, end_idx=du33+step,
                                    root_dir=data_dir, train_size=0.0, if_save_data=False)
            TOTAL = len(val_data2)
            val_iterator2 = DataLoader(val_data2, batch_size=64, shuffle=False, collate_fn=sim_collate_fn)
            res = eval_f(sae, val_iterator2, TOTAL, device)
            corr_ious.extend(res[0])
            wrong_ious.extend(res[1])
            all_wrong.extend(res[2])
            print(du33, du33+step, np.mean(corr_ious), np.mean(wrong_ious))
            val_data2.clear()
            del val_data2
            del val_iterator2
            gc.collect()

        with open("data/corr.iou", 'wb') as f:
            pickle.dump(corr_ious, f, pickle.HIGHEST_PROTOCOL)
        with open("data/wrong.iou", 'wb') as f:
            pickle.dump(wrong_ious, f, pickle.HIGHEST_PROTOCOL)