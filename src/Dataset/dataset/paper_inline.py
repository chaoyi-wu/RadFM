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

class Paper_Inline_dataset(Dataset):
    """
    Dataset class for processing scientific papers with inline images.
    
    This dataset extracts text and associated images from scientific papers,
    preparing them for multimodal model training.
    """
    def __init__(self, csv_path, img_path, sample_sentence_length=50, max_img_size=3):
        """
        Initialize the dataset.
        
        Args:
            csv_path: Path to CSV file containing paper metadata
            img_path: Root directory for paper figures
            sample_sentence_length: Maximum number of sentences to include in a sample
            max_img_size: Maximum number of images to include in a sample
        """
        self.max_img_size = max_img_size
        self.sample_sentence_length = sample_sentence_length
        self.img_path = img_path
        # Load paper paths from CSV
        self.paper_path = np.array(pd.read_csv(csv_path)['PMC_path'])
        
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
        """Return the total number of papers in the dataset"""
        return self.paper_path.shape[0]
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset
        
        Args:
            idx: Index of the paper to retrieve
            
        Returns:
            Dictionary containing the processed sample with images, question, and answer
        """
        # Load the paper JSON file
        paper_json = self.paper_path[idx]
        # Extract PMC ID from the file path
        PMC_name = paper_json.rsplit('/', 2)[-1].split('.')[0]
        # Load the list of sentences with image references
        sentences_list = json.load(open(paper_json, 'r'))
        # Process the paper to extract text and images
        image_dict, question, answer = self.random_sample_sentence(sentences_list, PMC_name)
        
        # Return formatted sample
        # Note: question is empty since this is for pretraining with full paper text
        return {
            "image_dict": image_dict,  # List of images with position information
            "question": question,      # Empty string for this dataset
            "answer": answer,          # Full text content
            }

    def random_sample_sentence(self, sentences_list, PMC_name):
        """
        Sample a segment of sentences from a paper and process inline images
        
        Args:
            sentences_list: List of sentences with image references
            PMC_name: PubMed Central ID for the paper
            
        Returns:
            Tuple of (processed_images, question_text, answer_text)
        """
        sentences_length = len(sentences_list)
        
        # Select a segment of the paper - either randomly or around image references
        p = random.random()
        if p >= 0.5:
            # Random segment selection
            if len(sentences_list) > self.sample_sentence_length:
                start = random.randint(0, sentences_length - self.sample_sentence_length)
                sentences_list = sentences_list[start:(start + self.sample_sentence_length)]
        else:
            # Try to select a segment containing images
            if len(sentences_list) > self.sample_sentence_length:
                sample_start = []
                # Find sentences with image references
                for sentence_id in range(len(sentences_list)):
                    if sentences_list[sentence_id]['img_ref'] != []:
                        # Start 10 sentences before the image if possible
                        if sentence_id - 10 < 0:
                            sample_start.append(0)
                        else:
                            if sentence_id - 10 > sentences_length - self.sample_sentence_length:
                                sample_start.append(sentences_length - self.sample_sentence_length)
                            else:
                                sample_start.append(sentence_id - 10)
                
                # If no images found, select random segment
                if sample_start == []:
                    start = random.randint(0, sentences_length - self.sample_sentence_length)
                    sentences_list = sentences_list[start:(start + self.sample_sentence_length)]
                else:
                    # Select a random segment that contains images
                    start = sample_start[random.randint(0, len(sample_start) - 1)]
                    sentences_list = sentences_list[start:(start + self.sample_sentence_length)]
            
        # Process the selected segment
        text = ''
        images = []
        for ix in sentences_list:
            sentence = ix
            if sentence["img_ref"] == []:
                # Add plain text without images
                text = text + sentence['text']
            else:
                # Stop if we've reached the maximum number of images
                if len(images) + len(sentence["img_ref"]) > self.max_img_size:
                    break
                    
                # Process each image referenced in the sentence
                for img_id in sentence["img_ref"]:
                    img_path = self.img_path + '/' + PMC_name + '_' + img_id + '.jpg'
                    if os.path.exists(img_path):
                        try:
                            # Load and transform the image
                            image = Image.open(img_path).convert('RGB')   
                            image = self.transform(image)
                            # Add image with position information
                            images.append({'image': image, "position": {"answer": len(text)}})    
                        except:
                            # Skip images that can't be loaded
                            continue
                # Add the text after processing images
                text = text + sentence['text']            
        
        # For this dataset, we don't use a question-answer format
        # Instead, all text is in the "answer" field
        question = ''
        answer = text
        
        return images, question, answer

# Example usage (commented out)
# csv_path = '/home/cs/leijiayu/wuchaoyi/multi_modal/Data/train_paper.csv'    
# img_path = '/home/cs/leijiayu/data/all_images/figures/'
# dataset = multi_paper_dataset(csv_path, img_path)
# print(dataset[0])