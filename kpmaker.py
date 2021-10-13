import torch
import os
import dlib
import matplotlib.pyplot as plt
from cropped import feature_generator

seg_model = r'.\shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(seg_model)

root_dir_list = [r".\data\my_dataset\train", r".\data\my_dataset\test"]

for root_dir in root_dir_list:
    for i in os.listdir(root_dir):
        dir_path = os.path.join(root_dir, i)
        img_path = os.path.join(dir_path, os.listdir(dir_path)[0])
        img = plt.imread(img_path)*255
        img = img.astype('uint8')
        torch.save(feature_generator(img, predictor)[1], os.path.join(r'.\keypoint_folder',str(i)+".pt"))

