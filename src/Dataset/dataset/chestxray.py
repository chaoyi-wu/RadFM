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

class ChestXray_Dataset(Dataset):
    """_summary_
    Args:
        Dataset (_type_): caption task formulated as vqa task for Chestxray classification dataset
        csv_path (_type_): path to csv file
        img_root_dir (_type_): path to image root directory 
        prompt_json_file (_type_): path to json file containing caption prompts
    Output:
        Dict: {
            "image_dict": {"image": image, "position": {"question": 0}}, # image is a tensor of shape [c,w,h,d] [3,512,512,1], position is a dict, random choice of 0 or len(question)
            "question": question, # random choice of caption prompts
            "answer":answer, # caption
            }
    """
    def __init__(self,csv_path,prompt_json_file):
        data_info = pd.read_csv(csv_path)
        self.img_path_list = np.asarray(data_info['image_path'])
        self.answer_list = np.asarray(data_info['label'])
        self.transform = transforms.Compose([                        
                transforms.RandomResizedCrop([512,512],scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
            ])   
        with open(prompt_json_file, 'r') as f:
            self.caption_prompts = json.load(f)['caption_prompt']

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        try:
            image = Image.open(img_path).convert('RGB')   
            image = self.transform(image)
            image = image.unsqueeze(-1) # c,w,h,d
        except:
            image = np.random.randn(3,512,512,4)
        
        answer = self.answer_list[index]
        question = random.choice(self.caption_prompts)
        image_dict = [{
                "image": image,
                "position": {
                    "question": len(question)
                }
            }]
        return {
            "image_dict": image_dict,
            "question": question,
            "answer":answer,
            }
  
    
if __name__ == "__main__":
    test_dataset = ChestXray_Dataset(csv_path = '../data_csv/chestxray.csv',  
                            prompt_json_file = './cls_prompt.json')
    for i in range(10):
        test_data = test_dataset[i]
        print(test_data['image_dict'][0]['image'].shape) # [3,512,512,1]
        #需要确保所有的chestxray img_path都有图像
    



