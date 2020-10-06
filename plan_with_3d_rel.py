from data_loader import PBW_3D
from models import LocationBasedGenerator
import torch

device = "cuda:0"

test_data = PBW_3D()
sg2behind = test_data.json2im["sg2behind"]
print(len(sg2behind))

model = LocationBasedGenerator()
model.to(device)
model.load_state_dict(torch.load("pre_models/model-20200726-041935", map_location=device))
acc = 0
total = 0.0
for test_sample in test_data:
    sg, obj_names, front_masks, behind_masks, behind_objects,\
    sg_n1, sg_n2, im_name, front_masks_name, behind_masks_name = test_sample

    front_masks = front_masks.to(device)

    pred_front_masks = model.return_sg(front_masks, [front_masks_name])[0]
    for pred in pred_front_masks:
        if pred not in sg:
            continue

    acc += 1
    total += 1
    print(acc, total, acc/total)
