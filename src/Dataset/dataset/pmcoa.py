import csv
import json
import logging
import os
import re
import difflib
import sys
import torch
import random
from abc import abstractmethod
from itertools import islice
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from collections.abc import Mapping
from torch.utils.data import DataLoader
import PIL
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
from collections import defaultdict
from PIL import Image

class PMCOA_Dataset(Dataset):
    """_summary_
    Args:
        Dataset (_type_): caption task formulated as vqa task for PMC-OA dataset
        csv_path (_type_): path to csv file
        img_root_dir (_type_): path to image root directory, with columns [PMC_ID,Figure_path,Caption]
        prompt_json_file (_type_): path to json file containing caption prompts
    Output:
        Dict: {
            "image_dict": {"image": image, "position": {"question": 0}}, # image is a tensor of shape [c,w,h,d] [3,512,512,1], position is a dict, random choice of 0 or len(question)
            "question": question, # random choice of caption prompts
            "answer":answer, # caption
            }
    """
    def __init__(self,csv_path,img_root_dir,prompt_json_file):
        self.img_root_dir = img_root_dir
        data_info = pd.read_csv(csv_path)
        self.img_path_list = np.asarray(data_info['Figure_path'])
        self.caption_list = np.asarray(data_info['Caption'])
        # PMC_ID,Figure_path,Caption
        # normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.transform = transforms.Compose([                        
                transforms.RandomResizedCrop([512,512],scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                # normalize,
            ])   

        with open(prompt_json_file, 'r') as f:
            self.caption_prompts = json.load(f)['caption_prompt']
    

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        file_name = self.img_path_list[index]
        img_path = os.path.join(self.img_root_dir,file_name)
        image = Image.open(img_path).convert('RGB')   
        image = self.transform(image) # normalize to [0,1]
        image = image.unsqueeze(-1) # expand a dimension
        answer = self.caption_list[index]
        question = random.choice(self.caption_prompts)
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
        
if __name__ == "__main__":
    test_dataset = PMCOA_Dataset(csv_path = '../data_csv/pmcoa_image_caption_train.csv',  
                            img_root_dir = '/home/cs/leijiayu/data/PMCVQA/caption_T060_filtered_top4_sep_v0_subfigures',  
                            prompt_json_file = './caption_prompt.json')
    for i in range(10):
        test_data = test_dataset[i]
        print(test_data['image_dict'][0]['image'].shape) # [3,512,512,1]
    
    



