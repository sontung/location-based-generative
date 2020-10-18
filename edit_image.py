from PIL import Image
import numpy as np

im = Image.open("../corl-3d-rel.jpg")
arr = np.array(im)
print(arr.shape)

# im1 = Image.fromarray(arr[:135])
# im1.save("s1.png")
#
# im2 = Image.fromarray(arr[170:350])
# im2.save("s2.png")
#
# im3 = Image.fromarray(arr[380:])
# im3.save("s3.png")
id_list = []
regions = [[], [], []]
first_reg = -1
incremented = False
for i in range(536):
    print(np.sum(arr[i]), 255 * 901 * 3, first_reg)
    try:
        if np.sum(arr[i]) < 600000:
            id_list.append(i)
            regions[first_reg].append(i)
            incremented = False
        else:
            if not incremented:
                first_reg += 1
                incremented = True
    except IndexError:
        break

print(arr[regions[0]].shape)
res = np.vstack((
    arr[regions[0]],
    np.ones((7, 901, 3), dtype=np.uint8)*255,
    arr[regions[1]],
    np.ones((7, 901, 3), dtype=np.uint8)*255,
    arr[regions[2]],
))
im1 = Image.fromarray(res)
im1.save("s1.png")


