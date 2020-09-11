import json
import numpy as np
from PIL import Image
with open("/home/sontung/Downloads/coco/annotations/instances_val2017.json", 'r') as json_file:
    du = json.load(json_file)

sheep_im = {}
bear_im = {}
for ann in du["annotations"]:
    if ann["category_id"] == 20:
        if ann["image_id"] not in sheep_im:
            sheep_im[ann["image_id"]] = [ann]
        else:
            sheep_im[ann["image_id"]].append(ann)
    elif ann["category_id"] == 23:
        if ann["image_id"] not in bear_im:
            bear_im[ann["image_id"]] = [ann]
        else:
            bear_im[ann["image_id"]].append(ann)

sheep_im_id = [(sheep_im[du1][0]["image_id"], sheep_im[du1][0]["bbox"]) for du1 in sheep_im]
bear_im_id = [(bear_im[du1][0]["image_id"], bear_im[du1][0]["bbox"]) for du1 in bear_im]
print(sheep_im_id[:10], bear_im_id[:10])

id2filename = {}

for im in du["images"]:
    assert im["id"] not in id2filename
    id2filename[im["id"]] = im["file_name"]

im_dir, box = bear_im_id[24]
count = 10
full_arr = Image.open("/home/sontung/Downloads/coco/images/val2017/%s" % id2filename[im_dir])
print(np.array(full_arr).shape, box)
mask = np.array(full_arr)[
       int(box[1]): int(box[1]+box[3]),
       int(box[0]): int(box[0]+box[2])]
Image.fromarray(mask).show()
Image.fromarray(mask).save("data/trivial_masks/b%d.png" % count)