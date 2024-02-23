import os
import cv2
import csv
import json
import imageio

import pandas as pd
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
from pydicom import dcmread

def dcm_to_png(dcm_path,save_png_path):
    ds = dcmread(dcm_path)
    arr = ds.pixel_array
    img_array = arr.copy()
    cv2.normalize(arr, img_array, 0, 255, cv2.NORM_MINMAX)
    img_array = np.array(img_array,dtype='uint8')
    # img_array = cv2.resize(img_array, (512,512), interpolation = cv2.INTER_LINEAR)
    imageio.imwrite(save_png_path,img_array) 

def preprocess_csv(csv_path,data_dir,save_data_dir):
    data_info = pd.read_csv(csv_path)
    patient_file_list = data_info.iloc[:,0]
    img_file_list = data_info.iloc[:,2]
    for idx in tqdm(range(len(img_file_list))):
        patient_file = patient_file_list[idx]
        img_file = img_file_list[idx]
        img_path = os.path.join(data_dir,str(patient_file),str(img_file)+'.dicom')
        os.makedirs(os.path.join(save_data_dir,str(patient_file)), exist_ok=True)
        save_img_path = os.path.join(save_data_dir,str(patient_file),str(img_file)+'.png')
        dcm_to_png(img_path,save_img_path)


csv_path = './DATA/VinDr/VinDr-Mammo/1.0.0/breast-level_annotations.csv'
data_dir = './DATA/VinDr/VinDr-Mammo/1.0.0/images'
save_data_dir = './DATA/VinDr/VinDr-Mammo/process/images'
os.makedirs(save_data_dir, exist_ok=True)
preprocess_csv(csv_path,data_dir,save_data_dir)
