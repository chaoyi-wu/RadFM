import os
import cv2
import csv
import json
import subprocess
import pandas as pd
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from scipy import ndimage
from collections import defaultdict

def resize_array(array_list, shape_list):
    if len(array_list) == 0:
        return None
    # Get the median value of the c dimension
    c_values = [shape[3] for shape in shape_list]
    z = np.median(c_values)
    
    # Resize each array to the same size
    resized_arrays = []
    for array in array_list:
        resized_array = ndimage.zoom(array, (3/array.shape[0],512/array.shape[1], 512/array.shape[2], z/array.shape[3]), order=0)
        # print(resized_array.shape)
        if resized_array.shape[3] == z:
            resized_arrays.append(resized_array)
        else:
            if resized_array.shape[3] > z:
                resized_arrays.append(resized_array[:,:,:,:int(z)])
            else:
                resized_arrays.append(np.pad(resized_array, ((0,0),(0,0),(0,0),(0,int(z-resized_array.shape[3]))), 'constant', constant_values=0))
    # Convert the list of arrays to a numpy array
    resized_array = np.array(resized_arrays)
    
    return resized_array

def process_image_list(image_path_list):
    image_shape_list = []
    image_array_list = []
    for image_path in image_path_list:
        if os.path.exists(image_path) == False:
            continue
        elif image_path.split('.')[-1] == 'npy':
            image_array = np.load(image_path) #c,w,h,d
            try:
                image_array = cv2.resize(image_array,(512,512))
                if len(image_array.shape) == 2:
                    image_array = image_array[np.newaxis,:,:,np.newaxis]
                    # 1wh1 to 3wh1
                    image_array = np.concatenate([image_array,image_array,image_array],axis=0)
                elif len(image_array.shape) == 3:
                    #whc to cwh
                    image_array = image_array.transpose(2,0,1)[:,:,:,np.newaxis]
                    
                image_shape_list.append(image_array.shape)
                image_array_list.append(image_array)
            except:
                pass
        else:
            itk_image = sitk.ReadImage(image_path)
            image_array = sitk.GetArrayFromImage(itk_image) #c,w,h,d
            if image_array.shape[0] != 512:
                image_array = cv2.resize(image_array,(512,512))
            if len(image_array.shape) == 2:
                image_array = image_array[np.newaxis,:,:,np.newaxis]
                image_array = np.concatenate([image_array,image_array,image_array],axis=0)
            elif len(image_array.shape) == 3:
                image_array = image_array[np.newaxis,:,:,:]
                image_array = np.concatenate([image_array,image_array,image_array],axis=0)
            image_shape_list.append(image_array.shape)
            image_array_list.append(image_array)
    save_image_array = resize_array(image_array_list, image_shape_list)
    return save_image_array
    
    
def process_json_file(json_file,save_json_file,save_root_dir):
    if not os.path.exists(save_root_dir):
        os.makedirs(save_root_dir)
    with open(json_file, 'r') as f:
        data = json.load(f)
    data_len = len(data)
    for i in tqdm(range(data_len)):
        samples = data[i]['samples']
        for sample_i in tqdm(range(len(samples))):
            if samples[sample_i]['image'] == []:
                samples.pop(sample_i)
            else:
                image_path_list = samples[sample_i]['image']
                case_id = image_path_list[0].split('/')[-3]
                save_image_array = process_image_list(image_path_list)
                if save_image_array is not None:
                    save_image_path = os.path.join(save_root_dir, str(case_id)+'_'+str(sample_i)+'.npy')
                    np.save(save_image_path,save_image_array)
                    # 如果边处理边传到aws的话可以参考这一段
                    # save_aws_image_path = save_image_path.replace('/mnt/petrelfs/share_data/zhangxiaoman/DATA/','s3://zhangxiaoman_hdd_new_share/')
                    # os.system(f'aws s3 cp {save_image_path} {save_aws_image_path}  --endpoint-url=http://10.140.27.254')
                    # os.remove(save_image_path)
                    # data[i]['npy_path'] = save_aws_image_path
                    data[i]['samples']['npy_path'] = save_image_path
                    data[i]['samples']['image_size'] = save_image_array.shape
                else:
                    print(i,image_path_list)
        if len(samples) == 0:
           data.pop(i)
            
    with open(save_json_file, 'w') as f:
        json.dump(data, f,ensure_ascii=False,indent=4)
    

if __name__ == "__main__":
    json_file = '../processed_file/processed_jsons/processed_json_2023-11-18.json'
    save_json_file = '../processed_file/processed_jsons/processed_json_2023-11-18-npy.json'
    save_root_dir = '../processed_file/processed_images'

    process_json_file(json_file,save_json_file,save_root_dir)
    