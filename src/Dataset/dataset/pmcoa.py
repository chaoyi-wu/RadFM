# Import necessary libraries for data processing, image handling, and model interaction
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
    """
    Dataset for processing scientific figures and captions from PubMed Central Open Access (PMC-OA).
    
    This dataset formulates image captioning as a visual question answering task,
    where the model is prompted with a question about an image and should respond
    with an appropriate caption.
    
    Args:
        csv_path: Path to CSV file with columns [PMC_ID, Figure_path, Caption]
        img_root_dir: Path to image root directory containing figure images
        prompt_json_file: Path to JSON file containing caption prompts
        
    Output:
        Dict: {
            "image_dict": [{"image": image, "position": {"question": position}}], 
            # image is a tensor of shape [c,w,h,d] [3,512,512,1]
            # position is where to insert the image - either at start (0) or end of question
            "question": question, # randomly selected caption prompt
            "answer": answer, # original caption from the paper
        }
    """
    def __init__(self, csv_path, img_root_dir, prompt_json_file):
        """
        Initialize the dataset.
        
        Args:
            csv_path: Path to CSV file with figure metadata
            img_root_dir: Root directory containing figure images
            prompt_json_file: JSON file with caption prompts
        """
        self.img_root_dir = img_root_dir
        
        # Load metadata from CSV file
        data_info = pd.read_csv(csv_path)
        self.img_path_list = np.asarray(data_info['Figure_path'])
        self.caption_list = np.asarray(data_info['Caption'])
        
        # Define image transformation pipeline
        # normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.transform = transforms.Compose([                        
                # Crop and resize images to 512x512, maintaining 80-100% of original content
                transforms.RandomResizedCrop([512, 512], scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
                # Convert to tensor with values in [0, 1]
                transforms.ToTensor(),
                # normalize,  # Commented out normalization
            ])   

        # Load caption prompts from JSON file
        with open(prompt_json_file, 'r') as f:
            self.caption_prompts = json.load(f)['caption_prompt']
    

    def __len__(self):
        """Return the total number of samples in the dataset"""
        return len(self.img_path_list)

    def __getitem__(self, index):
        """
        Get a single sample from the dataset
        
        Args:
            index: Index of the sample to retrieve
            
        Returns:
            Dictionary containing processed sample with image, question prompt, and caption answer
        """
        # Get the image filename and construct full path
        file_name = self.img_path_list[index]
        img_path = os.path.join(self.img_root_dir, file_name)
        
        # Load and preprocess the image
        image = Image.open(img_path).convert('RGB')   
        image = self.transform(image)  # normalize to [0,1]
        image = image.unsqueeze(-1)  # add depth dimension [C, H, W, 1]
        
        # Get the caption and a random prompt
        answer = self.caption_list[index]
        question = random.choice(self.caption_prompts)
        
        # Randomly decide whether to place the image before or after the question
        if random.random() < 0.5:
            # Place image before the question
            image_dict = {
                "image": image,
                "position": {
                    "question": 0  # At the beginning of question
                }
            }
        else:
            # Place image after the question
            image_dict = {
                "image": image,
                "position": {
                    "question": len(question)  # At the end of question
                }
            }
            
        # Return formatted sample
        return {
            "image_dict": [image_dict],  # List containing one image with position info
            "question": question,         # Caption prompt
            "answer": answer,             # Ground truth caption
            }
        
if __name__ == "__main__":
    # Example usage for testing the dataset
    test_dataset = PMCOA_Dataset(
        csv_path='../data_csv/pmcoa_image_caption_train.csv',  
        img_root_dir='/home/cs/leijiayu/data/PMCVQA/caption_T060_filtered_top4_sep_v0_subfigures',  
        prompt_json_file='./caption_prompt.json'
    )
    
    # Test the first 10 samples
    for i in range(10):
        test_data = test_dataset[i]
        print(test_data['image_dict'][0]['image'].shape)  # Should print [3,512,512,1]