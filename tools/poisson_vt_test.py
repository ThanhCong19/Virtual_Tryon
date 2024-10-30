import os
import cv2
import numpy as np
from PIL import Image



path1 = r"D:\Capstone_Project\fptvton1\example\human\00055_00.jpg"
path2 = r"D:\Capstone_Project\img.png"

mask_path = os.path.join(r"D:\Capstone_Project\datasets\vitonhd\test\mask\00055_00.png")

src_path = os.path.join(path1)

dst_path = os.path.join(path2)

src = cv2.imread(src_path)
src = cv2.resize(src, (384, 512))

dst = cv2.imread(dst_path)

mask = Image.open(mask_path).convert("L").resize((384, 512))
mask = np.array(mask)
mask = 255-mask

output = cv2.seamlessClone(src, dst, mask, (192,256), cv2.NORMAL_CLONE)
cv2.imwrite("D:\Capstone_Project\output.png", output)
