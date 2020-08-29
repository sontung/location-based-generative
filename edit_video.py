

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

cap = cv2.VideoCapture("/home/sontung/Downloads/vid_rightTop_s1_0432_s2__s0_51.mp4")

c = 0
color = 'rgb(0, 0, 0)'  # black color

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    c+=1
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image = Image.fromarray(frame)
    font = ImageFont.truetype('/home/sontung/Downloads/Roboto/Roboto-Bold.ttf', size=45)
    image = image.resize((1920, 1080))
    draw = ImageDraw.Draw(image)

    (x, y) = (1200, 50)

    if c < 100:
        message = "pick %s and place on %s" % ("brown", "green")
    elif 100 <= c < 177:
        message = "pick %s and place on %s" % ("purple", "brown")
    elif 177 <= c < 250:
        message = "pick %s and place on %s" % ("pink", "stack2")
    elif 250 <= c < 330:
        message = "pick %s and place on %s" % ("purple", "red")
    elif 330 <= c < 403:
        message = "pick %s and place on %s" % ("brown", "purple")
    else:
        message = "pick %s and place on %s" % ("green", "pink")



    draw.text((x, y), message, fill=color, font=font)
    image.save("robo_vid/im-%d.jpeg" % c, subsampling=0)
    print(c)


print(c)
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()