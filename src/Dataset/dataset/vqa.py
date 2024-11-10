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

class VQA_Dataset(Dataset):
    """_summary_
    Args:
        Dataset (_type_):  
        csv_path (_type_): path to csv file
    Output:
        Dict: {
            "image_dict": {"image": image, "position": {"question": 0}}, # image is a tensor of shape [c,w,h,d] [3,512,512,1], position is a dict, random choice of 0 or len(question)
            "question": question, # random choice of caption prompts
            "answer":answer, # caption
            }
    """
    def __init__(self,csv_path):
        data_info = pd.read_csv(csv_path)
        self.img_root_dir_list = np.asarray(data_info['img_root_dir'])
        self.img_path_list = np.asarray(data_info['Figure_path'])
        self.question_list = np.asarray(data_info['Question'])
        self.answer_list = np.asarray(data_info['Answer'])
        # PMC_ID,Figure_path,Question,Answer
        self.transform = transforms.Compose([                        
                transforms.RandomResizedCrop([512,512],scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
            ])   


    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        file_name = self.img_path_list[index]
        img_root_dir = self.img_root_dir_list[index]
        img_path = os.path.join(img_root_dir,file_name)
        image = Image.open(img_path).convert('RGB')   
        image = self.transform(image)
        image = image.unsqueeze(-1)
        answer = self.answer_list[index]
        question = str(self.question_list[index])
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
    test_dataset = PMCVQA_Dataset(csv_path = '../data_csv/pmcvqa_train.csv')
    for i in range(10):
        test_data = test_dataset[i]
        print(test_data['image_dict'][0]['image'].shape) # [3,512,512,1]
    
    



