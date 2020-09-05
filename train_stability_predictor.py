import numpy as np
import skimage.draw
from itertools import product
from PIL import Image


def draw_circle(mat, cent, color):
    circle = skimage.draw.circle(cent[0], cent[1], 10)
    mat[circle] = color
    return mat, min(circle[0])

def draw_sq(mat, cent, color, size=10):
    square = skimage.draw.polygon([cent[0]-size, cent[0]+size, cent[0]+size, cent[0]-size],
                                  [cent[1]-size, cent[1]-size, cent[1]+size, cent[1]+size])
    mat[square] = color

    return mat, min(square[0])

def draw_tri(mat, cent, color, size=10):
    square = skimage.draw.polygon([cent[0]- size, cent[0] + size, cent[0] + size],
                                  [cent[1], cent[1] - size, cent[1] + size],
                                  )
    mat[square] = color
    return mat, min(square[0])

shape2im = {
    "circle": (draw_circle, [66, 128, 64]),
    "sq": (draw_sq, [194, 0, 64]),
    "tri": (draw_tri, [187, 187, 253])
}

def sg2im(stack):
    pos = 216
    image = np.zeros([224, 224, 3], dtype=np.uint8)
    for obj in stack:
        _, pos = shape2im[obj][0](image, [pos-11, 224/2], shape2im[obj][1])
    return image

def find_target(seq_):
    target_ = 0
    for idx_, i in enumerate(list(reversed(seq_))):
        if i in ["tri", "circle"]:
            target_ = idx_
    return target_

def generate_data(max_nb_objects=7):
    objects = ["circle", "tri", "sq"]
    im_id = 0
    for nb_objects in range(2, max_nb_objects+1):
        all_seq = list(product(objects, repeat=nb_objects))
        for seq in all_seq:
            im_id += 1
            target = find_target(seq)
            im_arr = sg2im(seq)
            im_name = "data/stability/im%d-stable%d.png" % (im_id, target)
            Image.fromarray(im_arr).save(im_name)

generate_data()

