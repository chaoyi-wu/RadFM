#processed cases accoring to case_id_list, and save a csv file, with image path and image caption
import os
import cv2
import csv
import json
import subprocess
import pandas as pd
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from collections import defaultdict

def get_image(single_image_dir,single_image_filenames):
    # single_image_filenames
    single_image_filenames.sort(key=lambda x: int(x.split('.')[0]))
    image_list = []
    for image_filename in single_image_filenames:
        image_file = os.path.join(single_image_dir, image_filename)
        #read jpeg to 2D array
        image_array = cv2.imread(image_file,0)
        if image_array is not None:
            image_size = image_array.shape
            image_array = cv2.resize(image_array,(512,512),interpolation = cv2.INTER_LINEAR)
            image_list.append(image_array)
        else:
            pass
    image_array = np.array(image_list) #c,w,h
    if len(image_array.shape) == 3:
        if image_array.shape[0] < image_array.shape[1]:
            image_array = image_array.transpose((1, 2, 0))
        # image_array = np.transpose(image_array, (2,0,1)) # w,h,c
    return image_array

gray_list = ['CT','MRI','X-ray','Ultrasound','Mammography']

def convert_case(case_id,image_root_dir,json_root_dir,save_case_dict,save_root_dir=None):
    # save_image_dir 
    case_images_dir = os.path.join(image_root_dir, case_id)
    case_json_path = os.path.join(json_root_dir, case_id+'.json')
    with open(case_json_path, 'r') as f:
        data = json.load(f)
    image_nums = (len(data.keys())-1)//2
    for image_num in range(1,image_nums+1):
        case_dict = defaultdict(list)
        image_dir = os.path.join(case_images_dir, str(image_num)) #./images/1/1
        image_caption = data[str(image_num) + '详情']
        image_modality = data[str(image_num)][0]['modality']
        
        single_image_names = os.listdir(image_dir)
        single_image_names.sort(key=lambda x: int(x.split('_')[1]))
        save_image_series = []
        
        for single_image_name in single_image_names:
            single_image_dir = os.path.join(image_dir, single_image_name)
            
            save_npy_dir = os.path.join(save_root_dir,str(case_id),str(image_num))
            
            
            single_image_filenames = os.listdir(single_image_dir)
            if len(os.listdir(single_image_dir)) == 1:
                # 2D image
                image_file = os.path.join(single_image_dir, single_image_filenames[0])
                save_image_array = cv2.imread(image_file) # w,h,c
            else:
                save_image_array = get_image(single_image_dir,single_image_filenames)
            if not os.path.exists(save_npy_dir):
                    os.makedirs(save_npy_dir)
            # print(save_image_array.shape)
            if save_image_array is not None:
                if len(save_image_array.shape) <=  5 and len(save_image_array.shape) >=2:
                    save_nii_path = os.path.join(save_npy_dir,single_image_name+'.nii.gz')
                    out = sitk.GetImageFromArray(save_image_array)
                    sitk.WriteImage(out, save_nii_path)
                    save_image_series.append(save_nii_path)
                else:
                    save_npy_path = os.path.join(save_npy_dir,single_image_name+'.npy')
                    np.save(save_npy_path,save_image_array)
                    save_image_series.append(save_npy_path)
        case_dict['image'] = save_image_series
        case_dict['image_caption'] = image_caption
        case_dict['image_modality'] = image_modality
        save_case_dict.append(case_dict)
    
if __name__ == "__main__":
    # case_id,image_root_dir,json_root_dir
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', default=0, type=int)
    parser.add_argument('--add_index', default=0, type=int)
    parser.add_argument('--start_index', default=1, type=int)
    parser.add_argument('--end_index', default=1000, type=int)
    args = parser.parse_args()
    
    image_root_dir = '/mnt/petrelfs/share_data/zhangxiaoman/DATA/Radio_VQA/processed_file/images'
    json_root_dir = '/mnt/petrelfs/share_data/zhangxiaoman/DATA/Radio_VQA/processed_file/jsons'
    save_root_dir = '/mnt/petrelfs/share_data/zhangxiaoman/DATA/Radio_VQA/processed_file/npys'
    save_case_dict = []
    
    args.start_index = args.index*1000+1 + args.add_index
    args.end_index = (args.index+1)*1000+1
    
    for case_id in tqdm(range(args.start_index,args.end_index)):
        case_id = str(case_id)
        convert_case(case_id,image_root_dir,json_root_dir,save_case_dict,save_root_dir)
        # CT_0 (200, 630, 630, 3)

    # save to csv
    save_json_file = '/mnt/petrelfs/share_data/zhangxiaoman/DATA/Radio_VQA/processed_file/processed_jsons/processed_json_'+str(args.index)+'.json'
    with open(save_json_file, 'w', encoding='utf-8') as f:
        json.dump(save_case_dict, f, ensure_ascii=False,indent=4)
    # B, S, T, W, H, Z
    # srun --partition=medai --mpi=pmi2 --quotatype=auto --gres=gpu:0 -n1 --ntasks-per-node=1  python data_convert.py --index 2 --add_index 24
    # cd /mnt/petrelfs/share_data/zhangxiaoman/DATA/Radio_VQA/jpeg2npy