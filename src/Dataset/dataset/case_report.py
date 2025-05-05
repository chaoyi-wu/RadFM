# Import necessary libraries for data processing, image handling, and model integration
from torch.utils.data import Dataset
import numpy as np
import transformers
import pandas as pd
import copy 
import random    
import os
import numpy as np
import tqdm
import torch
import json
from PIL import Image
import torchvision
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from torchvision import transforms
from ast import literal_eval

class CaseReport_dataset(Dataset):
    """
    Dataset class for medical case reports with associated images.
    
    This dataset processes medical case reports containing text and referenced images,
    formatting them for multimodal medical AI training or inference.
    """
    def __init__(self, csv_path, img_path):
        """
        Initialize the dataset.
        
        Args:
            csv_path: Path to CSV file containing case reports data
            img_path: Base path to the directory containing images
        """
        self.img_path = img_path  # Root directory for images
        self.question_list = pd.read_csv(csv_path)  # Load dataset from CSV
        
        # Define image transformation pipeline
        # normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.transform = transforms.Compose([                        
                # Crop and resize images to 512x512, maintaining 80-100% of original content
                transforms.RandomResizedCrop([512, 512], scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
                # Convert to tensor with values in [0, 1]
                transforms.ToTensor(),
                # normalize,  # Commented out normalization
            ])   
                
        
    def __len__(self):
        """Return the total number of samples in the dataset"""
        return len(self.question_list)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Dictionary containing the processed sample with image, question, and answer
        """
        # Get the row from dataframe
        sample = self.question_list.iloc[idx]
        
        # Extract metadata and content
        PMC_id = sample['PMC_id']  # PubMed Central ID
        img_ref = literal_eval(sample['img_ref'])  # List of image references
        context = str(sample['context'])  # Case context
        
        # Truncate long contexts to focus on beginning and end
        sentences = context.split('.')
        if len(sentences) > 5:
            first_sentence = sentences[0]  # Keep the first sentence
            last_sentences = ". ".join(context.split('.')[-4:])  # Keep the last 4 sentences
            context = first_sentence + '. ' + last_sentences
            
        # Format question by combining context and actual question
        question = str(context) + '\n' + str(sample['question']).replace('Q:', '') 
        
        # Clean up answer formatting
        answer = str(sample['answer']).replace('A:', '')
        
        # Process each referenced image
        images = []
        for img_id in img_ref:
            # Construct the full image path
            img_path = self.img_path + '/' + PMC_id + '_' + img_id + '.jpg'
            
            try:
                # Load and transform the image
                image = Image.open(img_path).convert('RGB')   
                image = self.transform(image)
                
                # Randomly decide where to place the image in the text
                # Either at the end of question or at the end of context
                if random.random() > 0.5:
                    images.append({'image': image, "position": {"question": len(question)}})    
                else:
                    images.append({'image': image, "position": {"question": len(context)}}) 
            except:
                # Skip images that can't be loaded
                continue        
    
        # Return formatted sample
        return {
            "image_dict": images,  # List of images with position information
            "question": question,  # Formatted question text
            "answer": answer,      # Answer text
            }

# Example usage (commented out)
# csv_path = '/gpfs/home/cs/leijiayu/wuchaoyi/multi_modal/Data/GPT_realdata/casa_report_train.csv'    
# img_path = '/home/cs/leijiayu/data/all_images/figures/'
# dataset = CaseReport_dataset(csv_path, img_path)
# print(dataset[0])