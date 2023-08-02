import csv
import json
import logging
import os
import re
import difflib
import sys
import cv2
import torch
import random
from abc import abstractmethod
from itertools import islice
from scipy import ndimage
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from collections.abc import Mapping
from torch.utils.data import DataLoader
import PIL
import SimpleITK as sitk
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
from collections import defaultdict
from PIL import Image
import math

class Radio_Modality_Dataset(Dataset):
    """_summary_
    Args:
        Dataset (_type_): _description_: modality asked task formulated as vqa task for Radiopaedia dataset
        csv_path (_type_): path to csv file
        prompt_json_file (_type_): path to json file containing caption prompts
    Output:
        Dict: {
             "image_dict": {"image": image, "position": {"question": 0}}, # image is a tensor of shape [s,c,w,h,d] like, [1,3,512,512,1], position is a dict, random choice of 0 or len(question)
            "question": question, # random choice of caption prompts
            "answer":answer, # caption
            }
    """
    def __init__(self,csv_path,prompt_json_file,modality_json_file,down_sample_ratio = 5):
        data_info = pd.read_csv(csv_path)
        self.down_sample_ratio = down_sample_ratio
        self.img_path_list = np.asarray(data_info['image_path'])
        self.caption_list = np.asarray(data_info['answer'])
        with open(prompt_json_file, 'r') as f:
            self.caption_prompts = json.load(f)['caption_prompt']
        with open(prompt_json_file, 'r') as f:
            self.modality_prompts = json.load(f)['modality_prompt']
        with open(modality_json_file, 'r') as f:
            self.modality_sets = json.load(f)['modality']
    
    def resize_image(self, image):
        if len(image.shape) == 3:
            if image.shape[0] > image.shape[2]:
                image = image.transpose(2,0,1)
            # print('before resize',image.shape)
            image = cv2.resize(image,(512,512),interpolation = cv2.INTER_LINEAR)
            # print('after resize',image.shape)
            image = image[np.newaxis,:,:,:]
            image = np.concatenate([image,image,image],axis=0)
        
        if image.shape[-1] > 64:
            image = ndimage.zoom(image, (3/image.shape[0],512/image.shape[1],512/image.shape[2],64/image.shape[3]), order=0)
        else:
            image = ndimage.zoom(image, (3/image.shape[0],512/image.shape[1],512/image.shape[2],1), order=0)
        return image

    def __len__(self):
        return math.ceil(len(self.img_path_list)/self.down_sample_ratio)
    
    def __getitem__(self, index):
        index = (self.down_sample_ratio*index +random.randint(0,self.down_sample_ratio-1))%len(self.img_path_list)
        img_path = self.img_path_list[index]
        try:
            itk_image = sitk.ReadImage(img_path)
            image = sitk.GetArrayFromImage(itk_image)
            image = self.resize_image(image)
        except:
            image = np.random.randn(3,512,512,4)
            
        # image = np.load(img_path) # c,w,h,d
        image = (image-image.min())/(image.max()-image.min())
        contain_nan = (True in np.isnan(image))
        if contain_nan:
            image = np.random.randn(3,512,512,4)
        image = torch.from_numpy(image).float()
        
        if random.random() < 0.5:
            #直接回答
            answer = self.caption_list[index]
            question = random.choice(self.caption_prompts)
        else:
            modality = self.caption_list[index]
            if random.random() < 0.5:
                    # 回答为yes
                question = random.choice(self.modality_prompts).replace('modality',modality)
                answer = 'yes'
            else:
                select_modality = modality
                while select_modality == modality:
                    select_modality = random.choice(list(self.modality_sets))
                question = random.choice(self.modality_prompts).replace('modality',modality)
                answer = 'no'
        if random.random() < 0.5:
            image_dict = {
                "image": image,
                "position": {
                    "question": 0
                }
            }
        else:
            image_dict = {
                "image": image,
                "position": {
                    "question": len(question)
                }
            }
        return {
            "image_dict": [image_dict],
            "question": question,
            "answer":answer,
            }

class RadioVQA_Dataset(Dataset):
    """_summary_

    Args:
        Dataset (_type_): _description_: caption task formulated as vqa task for Radiopaedia dataset
        csv_path (_type_): path to csv file
        prompt_json_file (_type_): path to json file containing caption prompts
    Output:
        Dict: {
             "image_dict": {"image": image, "position": {"question": 0}}, # image is a tensor of shape [s,c,w,h,d] like, [1,3,512,512,1], position is a dict, random choice of 0 or len(question)
            "question": question, # random choice of caption prompts
            "answer":answer, # caption
            }
    """
    def __init__(self,csv_path):
        data_info = pd.read_csv(csv_path)
        # npy_path,image_caption,question,answer
        self.img_path_list = np.asarray(data_info['image_path'])
        self.question_list = np.asarray(data_info['question'])
        self.answer_list = np.asarray(data_info['answer'])
    
    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        image = np.load(img_path)
            
        image = (image-image.min())/(image.max()-image.min())
        contain_nan = (True in np.isnan(image))
        if contain_nan:
            image = np.random.randn(3,512,512,4)

        image = torch.from_numpy(image).float()
        answer = self.answer_list[index]
        question = self.question_list[index]
        image_dict = []
        for idx in range(image.shape[0]):
            if random.random() < 0.5:
                dict_idx = {
                    "image": image[idx],
                    "position": {
                        "question": 0
                    }
                }
            else:
                dict_idx = {
                    "image": image[idx],
                    "position": {
                        "question": len(question)
                    }
                }
            image_dict.append(dict_idx)
        if len(image_dict) > 10:
            images = random.sample(image_dict,10) 
        return {
            "image_dict": image_dict,
            "question": question,
            "answer":answer,
            }

class RadioCaption_Dataset(Dataset):
    def __init__(self,json_path,prompt_json_file):
        with open(json_path, 'r') as file:
            self.json_data = json.load(file)
        with open(prompt_json_file, 'r') as f:
            self.caption_prompts = json.load(f)['caption_prompt']

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, index):
        data_index = self.json_data[index]
        patient_pre = data_index['pre']
        patient_pat = data_index['pat']
        img_path = data_index['npy_path']
        finding = data_index['finding']
        impression = data_index['impression']
        prompt_question = random.choice(self.caption_prompts)
        question = patient_pat + ' ' + patient_pre + ' ' + prompt_question
        image = np.load(img_path)
        image = (image-image.min())/(image.max()-image.min())
        contain_nan = (True in np.isnan(image))
        if contain_nan:
            image = np.random.randn(3,512,512,4)
        image = torch.from_numpy(image).float()
        answer = 'Finding: ' + str(finding) + 'Impression: ' + str(impression) 
        
        image_dict = []
        for idx in range(image.shape[0]):
            if random.random() < 0.5:
                dict_idx = {
                    "image": image[idx],
                    "position": {
                        "question": 0
                    }
                }
            else:
                dict_idx = {
                    "image": image[idx],
                    "position": {
                        "question": len(question)
                    }
                }
            image_dict.append(dict_idx)
        return {
            "image_dict": image_dict,
            "question": question,
            "answer":answer,
            } 


class Radiofeatures_Dataset(Dataset):
    def __init__(self,json_path,prompt_json_file,disease_prompt_json_file,article_json_file):
        with open(json_path, 'r') as file:
            self.json_data = json.load(file)
        with open(prompt_json_file, 'r') as f:
            self.caption_prompts = json.load(f)['caption_prompt']
        with open(disease_prompt_json_file, 'r') as f:
            self.disease_prompts = json.load(f)['caption_prompt']
        with open(article_json_file, 'r') as f:
            self.article_sets = json.load(f).keys()
            
    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, index):
        data_index = self.json_data[index]
        patient_pre = data_index['pre']
        patient_pat = data_index['pat']
        img_path = data_index['npy_path']
        radiographic_features = ' '.join(data_index['radiographic_features'])
        image = np.load(img_path)
        image = (image-image.min())/(image.max()-image.min())
        contain_nan = (True in np.isnan(image))
        if contain_nan:
            image = np.random.randn(3,512,512,4)
        image = torch.from_numpy(image).float()
        
        if random.random() < 0.5:
            articles = ' '.join(data_index['articles'])
            prompt_question = random.choice(self.caption_prompts)
            question = patient_pat + ' ' + patient_pre + ' ' + prompt_question
            answer = articles + 'The Radiographic features can be summarized as follows.' + radiographic_features
        else:
            articles = data_index['title']
            if random.random() < 0.5:
                    # 回答为yes
                question = random.choice(self.disease_prompts).replace('disease',articles)
                answer = 'yes'
            else:
                select_articles = articles
                while select_articles == articles:
                    select_articles = random.choice(list(self.article_sets))
                question = random.choice(self.disease_prompts).replace('disease',select_articles)
                answer = 'no'
        image_dict = []
        for idx in range(image.shape[0]):
            if random.random() < 0.5:
                dict_idx = {
                    "image": image[idx],
                    "position": {
                        "question": 0
                    }
                }
            else:
                dict_idx = {
                    "image": image[idx],
                    "position": {
                        "question": len(question)
                    }
                }
            image_dict.append(dict_idx)
            
        return {
            "image_dict": image_dict,
            "question": question,
            "answer":answer,
            } 

"""
class Radiodisease_Dataset(Dataset):
    def __init__(self,json_path,prompt_json_file,article_json_file):
        with open(json_path, 'r') as file:
            self.json_data = json.load(file)
        with open(prompt_json_file, 'r') as f:
            self.caption_prompts = json.load(f)['caption_prompt']
        with open(article_json_file, 'r') as f:
            self.article_sets = json.load(f).keys()
        
    def __len__(self):
        return len(self.json_data)
    
    def __getitem__(self, index):
        data_index = self.json_data[index]
        patient_pre = data_index['pre']
        patient_pat = data_index['pat']
        img_path = data_index['npy_path']
        articles = data_index['title']
        if random.random() < 0.5:
            # 回答为yes
            question = random.choice(self.caption_prompts).replace('disease',articles)
            answer = 'yes'
        else:
            select_articles = articles
            while select_articles == articles:
                select_articles = random.choice(list(self.article_sets))
            question = random.choice(self.caption_prompts).replace('disease',select_articles)
            answer = 'no'
        image = np.load(img_path)
        image = (image-image.min())/(image.max()-image.min())
        image = torch.from_numpy(image).float()
        
        image_dict = []
        for idx in range(image.shape[0]):
            if random.random() < 0.5:
                dict_idx = {
                    "image": image[idx],
                    "position": {
                        "question": 0
                    }
                }
            else:
                dict_idx = {
                    "image": image[idx],
                    "position": {
                        "question": len(question)
                    }
                }
            image_dict.append(dict_idx)
        return {
            "image_dict": image_dict,
            "question": question,
            "answer":answer,
            } 


class Radio_modality_binary_Dataset(Dataset):
    def __init__(self,csv_path,prompt_json_file,modality_json_file):
        data_info = pd.read_csv(csv_path)
        self.img_path_list = np.asarray(data_info['image_path'])
        self.caption_list = np.asarray(data_info['answer'])
        with open(prompt_json_file, 'r') as f:
            self.caption_prompts = json.load(f)['modality_prompt']
        with open(modality_json_file, 'r') as f:
            self.modality_sets = json.load(f)['modality']
            
    def resize_image(self, image):
        if len(image.shape) == 3:
            if image.shape[0] > image.shape[2]:
                image = image.transpose(2,0,1)
            # print('before resize',image.shape)
            image = cv2.resize(image,(512,512),interpolation = cv2.INTER_LINEAR)
            # print('after resize',image.shape)
            image = image[np.newaxis,:,:,:]
            image = np.concatenate([image,image,image],axis=0)
        
        if image.shape[-1] > 64:
            image = ndimage.zoom(image, (3/image.shape[0],512/image.shape[1],512/image.shape[2],64/image.shape[3]), order=0)
        else:
            image = ndimage.zoom(image, (3/image.shape[0],512/image.shape[1],512/image.shape[2],1), order=0)
        return image

    def __len__(self):
        return len(self.img_path_list)
    
    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        try:
            itk_image = sitk.ReadImage(img_path)
            image = sitk.GetArrayFromImage(itk_image)
            image = self.resize_image(image)
        except:
            image = np.random.randn(3,512,512,4)
        
        image = (image-image.min())/(image.max()-image.min())
        image = torch.from_numpy(image).float()
        modality = self.caption_list[index]
        
        if random.random() < 0.5:
            # 回答为yes
            question = random.choice(self.caption_prompts).replace('modality',modality)
            answer = 'yes'
        else:
            select_modality = modality
            while select_modality == modality:
                select_modality = random.choice(list(self.modality_sets))
            question = random.choice(self.caption_prompts).replace('modality',modality)
            answer = 'no'
        
        if random.random() < 0.5:
                image_dict = {
                "image": image,
                "position": {
                    "question": 0
                }
            }
        else:
            image_dict = {
                "image": image,
                "position": {
                    "question": len(question)
                }
            }
        return {
            "image_dict": [image_dict],
            "question": question,
            "answer":answer,
            }
"""