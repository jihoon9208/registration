import os
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import imageio


save_dir = './outdoor_image'
filenames = [i for i in os.listdir(save_dir) ] ## 필터링할 때 
with imageio.get_writer('./result.gif', mode='I',duration=0.5) as writer:
    filenames = sorted(filenames)
    for filename in filenames:
        filename = os.path.join(save_dir , filename)
        image = imageio.imread(filename)
        writer.append_data(image)