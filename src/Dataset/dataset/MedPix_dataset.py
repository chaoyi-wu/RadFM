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
import re
import math

class MedPix_Single_Dataset(Dataset):
    """
    Dataset class for single-image MedPix data.
    
    Processes single medical images with various prompts related to modality,
    plane orientation, and general image captioning.
    """
    def __init__(self, csv_path, img_root="/gpfs/home/cs/leijiayu/data/MedPix/images/", down_sample_ratio=5):
        """
        Initialize the dataset.
        
        Args:
            csv_path: Path to CSV file containing image metadata
            img_root: Root directory for images
            down_sample_ratio: Factor to reduce dataset size
        """
        self.case_list = pd.read_csv(csv_path)
        self.img_root = img_root
        # Image transformation pipeline
        self.transform = transforms.Compose([                        
                transforms.RandomResizedCrop([512, 512], scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                # normalize,  # Commented out normalization
            ])  
        self.down_sample_ratio = down_sample_ratio
        
        # Define template prompts for different tasks
        self.promt = {
            # Image captioning prompts
            "caption": [
                "Describe this input image.",
                "Help captioning the image.",
                "What can be inflected from the scan?",
                "Can you give a caption for this image?",
                "Can you provide a brief summary of the radiology image?",
                "Please write a report about the image?",
                "Can you provide an analysis of this image?",
                "Can you explain what is shown in this image?",
                "What can be indicated from the radiologic scans?",
                "What can you infer from this photograph?",
            ],
            # Modality identification prompts
            "modality": [
                "What is the modality of the image?",
                "What type of imaging technique was utilized?",
                "What imaging technology was used?",
                "Please tell me the modality of the image.",
                "Describe the modality of the image.",
                "Tell me the imaging technology used.",
                "Can you specify the imaging modality used?",
                "What kind of imaging modality was applied?",
                "Which imaging technique was used for this image?",
                "Could you identify the imaging modality of this picture?",
                "What type of image modality was used here?",
                "Can you describe the imaging technique used?"
            ],
            # Plane orientation prompts
            "plane": [
                "Please distinguish the plane of the image",
                "Which view does this scan take from?",
                "Describe the position.",
                "What angle is this image taken from?",
                "Can you explain the orientation of this picture?",
                "From which direction was this shot taken?",
                "Can you specify the plane of this picture?",
                "From which standpoint is this image taken?",
                "Tell me which plane is the image.",
                "From what angle is this picture captured?",
                "Can you determine the shot direction of this image?",
                "Can you describe the plane of this image?",
            ],
            # Yes/no prompts for modality
            "modality_yes_no": [
                "Is this image shot in {object}?",
                "Is this image in {object}?",
                "Is {object} used fro this image?",
                "Was this picture taken in {object}?",
                "Was this photo captured in {object}?",
                "Did they use {object} for this image?",
                "Is this picture from {object}?",
                "Is this scan shot in {object}?"
            ],
            # Yes/no prompts for plane orientation
            "plane_yes_no": [
                "Is this image shot from {object} view?",
                "Is this image in the view of {object}?",
                "Was this scan in {object} view?",
                "Is this photo shot in {object} position?",
                "Was this picture taken from the perspective of {object}?",
                "Is this image captured from {object} viewpoint?",
                "Is this photograph from the angle of {object}?",
                "Is this snapshot from the view of {object}?",
            ],
        }
        
        # Lists of possible values for modality and plane categories
        self.sample_list = { 
                'modality': ['HE - High Power (>200X)', 'MR - FLAIR', 'Mammograph', 'SPECT', 
                              'MR - FLAIR w/Gd', 'UGI - Upper GI', 'OPHTH - Fundoscopy', 'SBFT - Small Bowel', 
                              'Special Stain (specify in caption)', 'EM - Electron Microscopic',
                              'MR T2* gradient GRE', 'CT - Montage', 'ECG EKG', 'MR - T2 FLAIR w/Contrast', 
                              'CT - noncontrast', 'MR - ADC Map (App Diff Coeff)', 'Interventional Procedure', 
                              'BE - Barium Enema', 'HE - Low Power (<50x)', 'MR - T2 weighted', 'MR - T1W w/Gd (fat suppressed)', 
                              'AN - Angiogram', 'OR - Operative photograph', 'Montage of Images', 'XR - Plain Film', 
                              'MR - T1W - noncontrast', 'BAS - Barium Swallow', 'US - Ultrasound', 'LOGO', 
                              'HE - Med Power (~50-200x)', 'NM - Nuclear Medicine', 'GR - Gross photograph', 
                              'MR - Other Pulse Seq.', 'Dermatology', 'IVP/IVU - Intravenous Urogram/Pyelogram', 
                              'VCUG - Voiding Cystourethrogram', 'CT - GI Contrast', 'MRS - Spectroscopy', 'MR - Montage', 
                              'Photograph', 'MRA - MR Angiography/Venography', 'MR - T1W w/Gadolinium', 'HSG - Hysterosalpingogram', 
                              'MR T2* gradient,GRE,MPGR,SWAN,SWI', 'Histology - Special Stain (specify in caption)', 'Venogram', 
                              'Arthrogram', 'CT - Myelogram', 'US-D - Doppler Ultrasound', 'CT - GI & IV Contrast', 
                              'CP - Clinical photograph', 'Histology (NOS)', 'Not Assigned', 'MR - PDW Proton Density', 
                              'CT w/contrast (IV)', 'OPHTH - Slit-Lamp', 'CTA - CT Angiography', 'AN - Angiogram (Catheter)', 
                              'MR - T1W SPGR', 'Tomography', 'EP - Endoscopy', 'PET-CT Fusion', 'MR - DWI Diffusion Weighted', 
                              'Drawing', 'PET - Positron Emission', 'SPECT - Single Photon', 'RU - Retrograde Urogram', 
                              'Myelogram', 'Fundoscopy', 'Virtual Colonoscopy', 'Photographs', 
                              'Interventional Procedure (specify in caption)', 'MR - STIR', 'MR - FIESTA'],
                'plane':    ['Other View (see caption)', 
                                'Mammo - CC', 'Sagittal', 'Image Plane', 'Mammo - XCC', 'Lateral', 'Longitudinal', 
                                'Mammo - Mag CC', 'Frontal', 'Mammo - MLO', 'Transverse', 'Gross Pathology', 'Dermatology', 
                                '3D Reconstruction', 'Photograph', 'Histology', 'PA', 'Decubitus', 'Multiple or Montage', 
                                'Oblique', 'AP', 'Drawing', 'Axial', 'Coronal'],
            }
        
        
    def __len__(self):
        """Return effective length of dataset after downsampling"""
        return math.ceil(len(self.case_list)/self.down_sample_ratio)
    
    def get_image(self, img_path):
        """
        Load and preprocess an image
        
        Args:
            img_path: Path to the image file
            
        Returns:
            Processed image tensor with shape [C, H, W, 1]
        """
        image = Image.open(img_path).convert('RGB')   
        image = self.transform(image)
        image = image.unsqueeze(-1)  # Add depth dimension [C, H, W, 1]
        return image
    
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Dictionary containing processed sample with image, question, and answer
        """
        # Apply downsampling with random offset
        idx = (self.down_sample_ratio*idx + random.randint(0, self.down_sample_ratio-1)) % len(self.case_list)
        sample = self.case_list.iloc[idx]
        answer = sample['context']
        
        # Handle different question types
        if sample['type'] == "modality" or sample['type'] == "plane":
            pp = random.random()
            if pp > 0.5:
                # Direct question about modality or plane
                question = random.sample(self.promt[sample['type']], 1)[0]
            else:
                # Yes/no question about modality or plane
                question = random.sample(self.promt[sample['type']+'_yes_no'], 1)[0]
                ppp = random.random()
                if ppp > 0.5:
                    # True case - format question with correct attribute
                    question = question.format(object=answer)
                    answer = 'yes'
                else:
                    # False case - randomly select a different attribute
                    sample_list = self.sample_list[sample['type']]
                    try:
                        sample_list.remove(answer)
                    except:
                        pass
                    answer = random.sample(sample_list, 1)[0]
                    question = question.format(object=answer)
                    answer = 'no'        
        else:
            # For other types, just select a random prompt
            question = random.sample(self.promt[sample['type']], 1)[0]
            
        # Randomly decide where to position the image - before or after question
        p = random.random()
        images = []
        if p > 0.5:
            try:
                # Place image after question
                images.append(
                    {
                        "image": self.get_image(self.img_root+sample['name']),
                        "position": {
                            "question": len(question)
                        }
                    }
                )   
            except:
                pass
        else:
            try:
                # Place image before question
                images.append(
                    {
                        "image": self.get_image(self.img_root+sample['name']),
                        "position": {
                            "question": 0 
                        }
                    }
                )   
            except:
                pass   
                
        # Return formatted sample
        return {
            "image_dict": images,
            "question": str(question),
            "answer": str(answer),
            }

class MedPix_Multi_Dataset(Dataset):
    """
    Dataset class for multi-image MedPix data.
    
    Processes cases with multiple medical images and supports various 
    diagnostic and analytical prompts.
    """
    def __init__(self, csv_path, img_root="/gpfs/home/cs/leijiayu/data/MedPix/images/"):
        """
        Initialize the dataset.
        
        Args:
            csv_path: Path to CSV file containing case metadata
            img_root: Root directory for images
        """
        self.case_list = pd.read_csv(csv_path)
        self.img_root = img_root
        # Image transformation pipeline
        self.transform = transforms.Compose([                        
                transforms.RandomResizedCrop([512, 512], scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                # normalize,  # Commented out normalization
            ]) 
        
        # Define template prompts for different diagnostic tasks
        self.promt = {
            # Treatment and follow-up prompts
            "txFollowup": [
                "What treatment should the patient take?",
                "Please give me some treatment advise.",
                "What is the recommended treatment for this condition?",
                "What kind of treatment is necessary for this patient?",
                "Can you suggest a suitable treatment for this case?",
                "What treatment options are available for this patient?",
                "What is the best course of treatment for this condition?",
                "How to follow up with the patient?",
                "What treatment should be administered for this illness?",
                "What is the most effective treatment for this disease?"
            ],
            # Differential diagnosis prompts
            "ddx": [
                "What illness can you diagnose from this images?",
                "What disease is shown in the scans?",
                "Please make diagnosis with the input images?",
                "What health issue can be inferred from these photos?",
                "What is the diagnosis based on these medical scans?",
                "Based on these scans, what is the patient suffering from?",
                "What ailment can be deduced from these medical images?",
                "Can you determine the illness from these medical photos?",
                "Can you identify the disease from these scans?",
                "What is the medical diagnosis based on these images?",
            ],
            # Diagnostic method prompts
            "dxHow": [
                "What imaging technology is used for diagnosis?",
                "What imaging tests are shown in the images?",
                "What type of imaging technique is used in medical diagnosis?",
                "What kind of imaging technology is used for medical purposes?",
                "Which imaging tests are depicted in these pictures?",
                "Can you identify the imaging tests in these images?",
                "What kind of imaging technology is used in healthcare?",
                "What imaging procedures are used for diagnosing diseases?",
                "Can you name the imaging tests shown in these photographs?",
                "Please distinguish the imaging type in these images",
            ],
            # General diagnosis prompts
            "diagnosis": [
                "What condition can be diagnosed from these pictures?",
                "Can you interpret the disease from these medical scans?",
                "What medical condition is depicted in these images?",
                "Based on these images, what could be the potential diagnosis?",
                "What health condition can be identified from these scans?",
                "Can you diagnose the disease from these medical images?",
                "What is the patient's condition according to these scans?",
                "What medical issue can be determined from these photos?",
                "Can you infer the illness from these medical scans?",
                "What is the probable diagnosis based on these medical images?",
                "What illness can you diagnose from this images?",
                "What disease is shown in the scans?",
                "Please make diagnosis with the input images?",
                "What health issue can be inferred from these photos?",
                "What is the diagnosis based on these medical scans?",
                "Based on these scans, what is the patient suffering from?",
                "What ailment can be deduced from these medical images?",
            ], 
            # Findings description prompts
            "findings": [
                "Caption the case.",
                "Describe your findings for this patient.",
                "What is shown in the case?",
                "Please help me write a report about the patient.",
                "Can you provide a summary of the case?",
                "What are the key points in this case?",
                "Could you explain the details of the case?",
                "What are your observations about the case?",
                "Can you give an overview of the case?",
                "How would you interpret this case?",
                "What is your analysis of the patient?",
                "Can you provide a brief on the patient?"
            ],
            # Exam result prompts
            "exam": [
                "Make a conclusion for this patient.",
                "What are the exam results for this patient?",
                "What is the diagnosis for this patient?",
                "What are the symptoms presented by this patient?",
                "Please make diagnosis with the input case.",
                "Is there any abnormality with the presented case?",
                "What can be reflected from the input images?",
                "Please provide me with some diagnosis advise.",
                "Can you provide a summary of the patient's condition?",
                "Can you provide a detailed analysis of the patient's condition?"
            ],
            # Case discussion prompts
            "discussion": [
                "Discuss about the case more.",
                "Tell more about the patient's illness.",
                "What image patterns or knowledge can help you make diagnosis?",
                "Could you provide more details about the situation?",
                "What additional information can you provide about the issue?",
                "Can you explain more about the subject matter?",
                "What other factors should be considered in this scenario?",
                "Can you provide more context or background information?",
                "What other relevant details can you share about this case?", 
                "Can you expand on your initial explanation?" ,
                "What other insights can you provide on this matter?" ,
                "Can you delve deeper into the specifics of the situation?",
            ],
        }  
        
    def __len__(self):
        """Return the total number of cases in the dataset"""
        return len(self.case_list)
    
    def get_image(self, img_path):
        """
        Load and preprocess an image
        
        Args:
            img_path: Path to the image file
            
        Returns:
            Processed image tensor with shape [C, H, W, 1]
        """
        image = Image.open(img_path).convert('RGB')   
        image = self.transform(image)
        image = image.unsqueeze(-1)  # Add depth dimension [C, H, W, 1]
        return image
    
    
    def __getitem__(self, idx):
        """
        Get a single case from the dataset
        
        Args:
            idx: Index of the case to retrieve
            
        Returns:
            Dictionary containing processed case with images, question, and answer
        """
        sample = self.case_list.iloc[idx]
        
        # Clean up answer text by removing bullet points
        answer = str(sample['context']).replace('• ', '')
        
        # Select random prompt for the specific task type
        question = random.sample(self.promt[sample['type']], 1)[0]
        
        # Optionally prepend patient history to the question
        history = sample['history']
        if history is not None:
            p = random.random()
            if p > 0.5:
                try:
                    question = history + ' ' + question
                except:
                    pass
                    
        # Process all images associated with this case
        image_names = sample['name'].split(',')
        p = random.random()
        images = []
        
        # Randomly decide whether to put images after or before question
        if p > 0.5:
            # Place images after question
            for pp in image_names:
                try:
                    images.append(
                        {
                            "image": self.get_image(self.img_root+pp),
                            "position": {
                                "question": len(question)
                            }
                        }
                    )    
                except:
                    pass
        else:
            # Place images before question
            for pp in image_names:
                try:
                    images.append(
                        {
                            "image": self.get_image(self.img_root+pp),
                            "position": {
                                "question": 0
                            }
                        }
                    ) 
                except:
                    pass
                    
        # For findings, remove measurements which might be distracting
        if sample['type'] == "findings":
            pattern = r"\d+(\.\d+)?\s*(mm|cm|x\d+\s*cm)"
            answer = re.sub(pattern, "", answer)
            
        # Limit number of images to prevent memory issues
        if len(images) > 10:
            images = random.sample(images, 10)
            
        # Return formatted case
        return {
            "image_dict": images,
            "question": str(question),
            "answer": str(answer),
            }

class MedPix_QA_Dataset(Dataset):
    """
    Dataset class for MedPix question-answer pairs.
    
    Processes medical QA pairs with associated images.
    """
    def __init__(self, csv_path, img_root="/gpfs/home/cs/leijiayu/data/MedPix/images/"):
        """
        Initialize the dataset.
        
        Args:
            csv_path: Path to CSV file containing QA pairs
            img_root: Root directory for images
        """
        self.case_list = pd.read_csv(csv_path)
        self.img_root = img_root
        # Image transformation pipeline
        self.transform = transforms.Compose([                        
                transforms.RandomResizedCrop([512, 512], scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                # normalize,  # Commented out normalization
            ]) 
                
    def __len__(self):
        """Return the total number of QA pairs in the dataset"""
        return len(self.case_list)
    
    def get_image(self, img_path):
        """
        Load and preprocess an image
        
        Args:
            img_path: Path to the image file
            
        Returns:
            Processed image tensor with shape [C, H, W, 1]
        """
        image = Image.open(img_path).convert('RGB')   
        image = self.transform(image)
        image = image.unsqueeze(-1)  # Add depth dimension [C, H, W, 1]
        return image
    
    
    def __getitem__(self, idx):
        """
        Get a single QA pair from the dataset
        
        Args:
            idx: Index of the QA pair to retrieve
            
        Returns:
            Dictionary containing processed QA pair with image, question, and answer
        """
        sample = self.case_list.iloc[idx]
        
        # Extract question, answer and explanation
        answer = sample['answer']
        question = sample['question']
        explanation = sample['explanation']
        
        # Combine answer with explanation when available
        try:
            answer = answer + '. ' + explanation
        except:
            pass
            
        # Randomly decide whether to place image before or after question
        p = random.random()
        images = []
        if p > 0.5:
            # Place image after question
            try:
                images.append(
                    {
                        "image": self.get_image(self.img_root+sample['name']),
                        "position": {
                            "question": len(question)
                        }
                    }
                )   
            except:
                pass
        else:
            # Place image before question
            try:
                images.append(
                    {
                        "image": self.get_image(self.img_root+sample['name']),
                        "position": {
                            "question": 0 
                        }
                    }
                )   
            except:
                pass  
                
        # Limit number of images to prevent memory issues
        if len(images) > 10:
            images = random.sample(images, 10) 
            
        # Return formatted QA pair
        return {
            "image_dict": images,
            "question": str(question),
            "answer": str(answer),
            }
                
# Example usage (commented out)
# dataset = MedPix_Single_Dataset(csv_path = '/gpfs/home/cs/leijiayu/data/MedPix/Preprocessor/MedPix_single_train.csv')
# for i in tqdm.tqdm(range(len(dataset))):
#     sample = dataset[i]
#     print(len(sample['image_dict']), sample['image_dict'][0]["image"].shape, sample['question'], sample['answer'])
#     input()

# dataset = MedPix_Multi_Dataset(csv_path = '/gpfs/home/cs/leijiayu/data/MedPix/Preprocessor/MedPix_multi_train.csv')
# for i in tqdm.tqdm(range(len(dataset))):
#     sample = dataset[i]
#     print(len(sample['image_dict']), sample['image_dict'][0]["image"].shape, sample['question'], sample['answer'])
#     input()
    
# dataset = MedPix_QA_Dataset(csv_path = '/gpfs/home/cs/leijiayu/data/MedPix/Preprocessor/MedPix_questions_train.csv')
# for i in tqdm.tqdm(range(len(dataset))):
#     sample = dataset[i]
#     print(len(sample['image_dict']), sample['image_dict'][0]["image"].shape, sample['question'], sample['answer'])
#     input()