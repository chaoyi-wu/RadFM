# Import necessary libraries for data processing and model interaction
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
import math
import torchvision
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from .dataset import *
import spacy
from spacy.tokens import Span
from scispacy.abbreviation import AbbreviationDetector
from scispacy.umls_linking import UmlsEntityLinker

class umls_extractor:
    """
    Extracts medical entities from text using scispacy and UMLS
    """
    def __init__(self):
        """Initialize the NLP pipeline with medical entity linking capabilities"""
        nlp = spacy.load("en_core_sci_lg")
        nlp.add_pipe("abbreviation_detector")
        nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
        self.nlp = nlp
        
    def extract(self, text):
        """
        Extract medical entities from text
        
        Args:
            text: Input text to process
            
        Returns:
            Set of extracted entities
        """
        doc = self.nlp(text)
        ent_set = doc.ents
        return ent_set

def find_position(label, key_embeddings):
    """
    Creates a tensor of weights for loss calculation based on important keywords
    
    Args:
        label: Token IDs tensor for the target sequence
        key_embeddings: List of token ID tensors for important medical terms
        
    Returns:
        Tensor of weights where important terms get higher weight (3)
    """
    loss_reweight = torch.ones(label.shape)
    for i in range(len(label)):
        if label[i] == -100:  # Skip padding or ignored tokens
            loss_reweight[i] = 0
        else:
            for key_embedding in key_embeddings:
                if torch.equal(label[i:i+len(key_embedding)], key_embedding):
                    loss_reweight[i:i+len(key_embedding)] = 3  # Increase weight for important terms
    return loss_reweight

def stack_images(images):
    """
    Processes and stacks a list of images to create a batch
    with consistent dimensions
    
    Args:
        images: List of image tensors
        
    Returns:
        Stacked and resized tensor of images
    """
    target_H = 512
    target_W = 512
    target_D = 4
    if len(images) == 0:
        return torch.zeros((1, 3, target_H, target_W, target_D))
    
    MAX_D = 4
    D_list = list(range(4, 65, 4))
    
    # Find maximum depth among all images
    for ii in images:
        try:
            D = ii.shape[3]
            if D > MAX_D:
                MAX_D = D
        except:
            continue
            
    # Select optimal target depth
    for temp_D in D_list:
        if abs(temp_D - MAX_D) < abs(target_D - MAX_D):
            target_D = temp_D
    
    # Resize and stack all images
    stack_images = []
    for s in images:
        if len(s.shape) == 3:
            stack_images.append(torch.nn.functional.interpolate(s.unsqueeze(0).unsqueeze(-1), size=(target_H, target_W, target_D)))
        else:
            stack_images.append(torch.nn.functional.interpolate(s.unsqueeze(0), size=(target_H, target_W, target_D)))
    
    images = torch.cat(stack_images, dim=0)
    return images


class multi_dataset(Dataset):
    """
    A dataset class that combines multiple medical imaging datasets
    for training a multimodal model
    """
    def __init__(self, text_tokenizer, max_seq=2048, max_img_size=100, image_num=32, voc_size=32000):
        """
        Initialize the multimodal dataset
        
        Args:
            text_tokenizer: Tokenizer or path to tokenizer
            max_seq: Maximum sequence length
            max_img_size: Maximum number of images to process
            image_num: Number of image tokens per image
            voc_size: Vocabulary size for the tokenizer
        """
        self.text_tokenizer = text_tokenizer
        self.max_img_size = max_img_size
        self.image_num = image_num
        self.max_seq = max_seq
        self.voc_size = voc_size
        self.H = 512
        self.W = 512
        self.image_padding_tokens = []
        self.words_extract = umls_extractor()
        
        # Initialize tokenizer if path is provided
        if isinstance(self.text_tokenizer, str):
            self.text_tokenizer = LlamaTokenizer.from_pretrained(
                self.text_tokenizer,
            )
            
            # Add special tokens for image references
            special_token = {"additional_special_tokens": ["<image>", "</image>"]}
            for i in range(max_img_size):
                image_padding_token = ""
                for j in range(image_num):
                    image_token = "<image" + str(i * image_num + j) + ">"
                    image_padding_token = image_padding_token + image_token
                    special_token["additional_special_tokens"].append("<image" + str(i * image_num + j) + ">")
                self.image_padding_tokens.append(image_padding_token)
                
            self.text_tokenizer.add_special_tokens(
                special_token
            )
            self.text_tokenizer.pad_token_id = 0
            self.text_tokenizer.bos_token_id = 1
            self.text_tokenizer.eos_token_id = 2

        # Initialize empty lists for dataset tracking
        self.data_whole_2D = []
        self.data_whole_3D = []
        self.dataset_reflect = {}
        
        ### 2D datasets
        ### Pretrain datasets - currently commented out
        # paper_inline_dataset = Paper_Inline_dataset(csv_path = '/gpfs/home/cs/leijiayu/wuchaoyi/multi_modal/Data/paper_train.csv', 
        #                                    img_path = '/home/cs/leijiayu/data/all_images/figures/')
        # self.dataset_reflect['paper_inline_dataset'] = paper_inline_dataset
        # self.data_whole_2D = self.data_whole_2D +  [{'paper_inline_dataset':i} for i in range(len(paper_inline_dataset))]
        # print('paper_inline_dataset loaded')
        
        # pmcoa_dataset = PMCOA_Dataset(csv_path = '/gpfs/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/data_csv/pmcoa_image_caption_train.csv',  
        #                     img_root_dir = '/home/cs/leijiayu/data/PMCVQA/caption_T060_filtered_top4_sep_v0_subfigures',  
        #                     prompt_json_file = '/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/dataset/caption_prompt.json')
        # self.dataset_reflect['pmcoa_dataset'] = pmcoa_dataset
        # self.data_whole_2D = self.data_whole_2D +  [{'pmcoa_dataset':i} for i in range(len(pmcoa_dataset))]
        # print('pmcoa_dataset loaded')
        
        ### SFT (Supervised Fine-Tuning) datasets
        ### MedPix datasets
        medpix_multi_dataset = MedPix_Multi_Dataset(csv_path='/gpfs/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/data_csv/MedPix_multi_train.csv')
        self.dataset_reflect['medpix_multi_dataset'] = medpix_multi_dataset
        self.data_whole_2D = self.data_whole_2D + [{'medpix_multi_dataset': i} for i in range(len(medpix_multi_dataset))]
        print('medpix_multi_dataset loaded')
        
        medpix_single_dataset = MedPix_Single_Dataset(csv_path='/gpfs/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/data_csv/MedPix_single_train.csv')
        self.dataset_reflect['medpix_single_dataset'] = medpix_single_dataset
        self.data_whole_2D = self.data_whole_2D + [{'medpix_single_dataset': i} for i in range(len(medpix_single_dataset))]
        print('medpix_single_dataset loaded')
        
        medpix_qa_dataset = MedPix_QA_Dataset(csv_path='/gpfs/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/data_csv/MedPix_questions_train.csv')
        self.dataset_reflect['medpix_qa_dataset'] = medpix_qa_dataset
        self.data_whole_2D = self.data_whole_2D + [{'medpix_qa_dataset': i} for i in range(len(medpix_qa_dataset))]
        print('medpix_qa_dataset loaded')
        
        ### Chest X-ray datasets
        ### Caption datasets
        chestxray_caption_dataset = ChestXray_Dataset(
            csv_path='/gpfs/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/data_csv/mimic_caption_train.csv',  
            prompt_json_file='/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/dataset/report_prompt.json'
        )
        self.dataset_reflect['chestxray_caption_dataset'] = chestxray_caption_dataset
        self.data_whole_2D = self.data_whole_2D + [{'chestxray_caption_dataset': i} for i in range(len(chestxray_caption_dataset))]
        print('chestxray_caption_dataset loaded')
        
        ### Binary classification datasets
        chestxray_dataset_bn = Binary_Dataset(
            csv_path='/gpfs/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/data_csv/chestxray_balance_train_new.csv',  
            prompt_json_file='/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/dataset/yes_no_prompt.json'
        )
        self.dataset_reflect['chestxray_dataset_bn'] = chestxray_dataset_bn
        self.data_whole_2D = self.data_whole_2D + [{'chestxray_dataset_bn': i} for i in range(len(chestxray_dataset_bn))]
        print('chestxray_dataset_bn loaded')
        
        pcxr_dataset_bn = Binary_Dataset(
            csv_path='/gpfs/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/data_csv/pcxr_balance_train.csv',  
            prompt_json_file='/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/dataset/yes_no_prompt.json'
        )
        self.dataset_reflect['pcxr_dataset_bn'] = pcxr_dataset_bn
        self.data_whole_2D = self.data_whole_2D + [{'pcxr_dataset_bn': i} for i in range(len(pcxr_dataset_bn))]
        print('pcxr_dataset_bn loaded')
        
        mammo_dataset_bn = Binary_Dataset(
            csv_path='/gpfs/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/data_csv/mammo_balance_train.csv',  
            prompt_json_file='/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/dataset/yes_no_prompt.json'
        )
        self.dataset_reflect['mammo_dataset_bn'] = mammo_dataset_bn
        self.data_whole_2D = self.data_whole_2D + [{'mammo_dataset_bn': i} for i in range(len(mammo_dataset_bn))]
        print('mammo_dataset_bn loaded')
        
        spinexr_dataset_bn = Binary_Dataset(
            csv_path='/gpfs/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/data_csv/spinexr_balance_train.csv',  
            prompt_json_file='/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/dataset/yes_no_prompt.json'
        )
        self.dataset_reflect['spinexr_dataset_bn'] = spinexr_dataset_bn
        self.data_whole_2D = self.data_whole_2D + [{'spinexr_dataset_bn': i} for i in range(len(spinexr_dataset_bn))]
        print('spinexr_dataset_bn loaded')
        
        ### Multi-label classification datasets
        chestxray_dataset = ChestXray_Dataset(
            csv_path='/gpfs/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/data_csv/chestxray_new.csv',  
            prompt_json_file='/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/dataset/cls_prompt.json'
        )
        self.dataset_reflect['chestxray_dataset'] = chestxray_dataset
        self.data_whole_2D = self.data_whole_2D + [{'chestxray_dataset': i} for i in range(len(chestxray_dataset))]
        print('chestxray_dataset loaded')
        
        pcxr_dataset = ChestXray_Dataset(
            csv_path='/gpfs/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/data_csv/pcxr_train_new.csv',  
            prompt_json_file='/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/dataset/cls_prompt.json'
        )
        self.dataset_reflect['pcxr_dataset'] = pcxr_dataset
        self.data_whole_2D = self.data_whole_2D + [{'pcxr_dataset': i} for i in range(len(pcxr_dataset))]
        print('pcxr_dataset loaded')
        
        mammo_dataset = ChestXray_Dataset(
            csv_path='/gpfs/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/data_csv/mammo_train_new.csv',  
            prompt_json_file='/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/dataset/mammo_prompt.json'
        )
        self.dataset_reflect['mammo_dataset'] = mammo_dataset
        self.data_whole_2D = self.data_whole_2D + [{'mammo_dataset': i} for i in range(len(mammo_dataset))]
        print('mammo_dataset loaded')
        
        spinexr_dataset = ChestXray_Dataset(
            csv_path='/gpfs/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/data_csv/spinexr_train_new.csv',  
            prompt_json_file='/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/dataset/spinexr_prompt.json'
        )
        self.dataset_reflect['spinexr_dataset'] = spinexr_dataset
        self.data_whole_2D = self.data_whole_2D + [{'spinexr_dataset': i} for i in range(len(spinexr_dataset))]
        print('spinexr_dataset loaded')
        
        ### VQA (Visual Question Answering) datasets
        pmcvqa_dataset = VQA_Dataset(csv_path='/gpfs/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/data_csv/pmcvqa_train.csv')
        self.dataset_reflect['pmcvqa_dataset'] = pmcvqa_dataset
        self.data_whole_2D = self.data_whole_2D + [{'pmcvqa_dataset': i} for i in range(len(pmcvqa_dataset))]
        print('pmcvqa_dataset loaded')
        
        casereport_dataset = CaseReport_dataset(
            csv_path='/gpfs/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/data_csv/filtered_case_report_train.csv',    
            img_path='/home/cs/leijiayu/data/all_images/figures/'
        )
        self.dataset_reflect['casereport_dataset'] = casereport_dataset
        self.data_whole_2D = self.data_whole_2D + [{'casereport_dataset': i} for i in range(len(casereport_dataset))]
        print('casereport_dataset loaded')
        
        vqarad_dataset = VQA_Dataset(csv_path='/gpfs/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/data_csv/vqarad_train.csv')
        self.dataset_reflect['vqarad_dataset'] = vqarad_dataset
        self.data_whole_2D = self.data_whole_2D + [{'vqarad_dataset': i} for i in range(len(vqarad_dataset))]
        print('vqarad_dataset loaded')
        
        slake_dataset = VQA_Dataset(csv_path='/gpfs/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/data_csv/slakevqa_train.csv')
        self.dataset_reflect['slake_dataset'] = slake_dataset
        self.data_whole_2D = self.data_whole_2D + [{'slake_dataset': i} for i in range(len(slake_dataset))]
        print('slake_dataset loaded')
        
        ### 3D datasets
        radiovqa_dataset = RadioVQA_Dataset(csv_path='/gpfs/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/data_csv/radiology_vqa_train.csv')
        self.dataset_reflect['radiovqa_dataset'] = radiovqa_dataset
        self.data_whole_3D = self.data_whole_3D + [{'radiovqa_dataset': i} for i in range(len(radiovqa_dataset))]
        print('radiovqa_dataset loaded')
        
        radiomodality_dataset = Radio_Modality_Dataset(
            csv_path='/gpfs/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/data_csv/radio_modality_train.csv',  
            prompt_json_file='/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/dataset/modality_prompt.json',
            modality_json_file='/gpfs/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/data_csv/modality_set.json'
        )
        self.dataset_reflect['radiomodality_dataset'] = radiomodality_dataset
        self.data_whole_3D = self.data_whole_3D + [{'radiomodality_dataset': i} for i in range(len(radiomodality_dataset))]
        print('radiomodality_dataset loaded')
        
        radiocaption_dataset = RadioCaption_Dataset(
            json_path='/gpfs/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/data_csv/radiology_article_npy_train.json',
            prompt_json_file='/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/dataset/caption_prompt.json',
        )
        self.dataset_reflect['radiocaption_dataset'] = radiocaption_dataset
        self.data_whole_3D = self.data_whole_3D + [{'radiocaption_dataset': i} for i in range(len(radiocaption_dataset))]
        print('radiocaption_dataset loaded')
        
        radiofeatures_dataset = Radiofeatures_Dataset(
            json_path='/gpfs/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/data_csv/radiology_article_npy_train.json',
            prompt_json_file='/gpfs/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/dataset/radiology_feature_prompt.json',
            disease_prompt_json_file='/gpfs/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/dataset/yes_no_prompt.json',
            article_json_file='/gpfs/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/data_csv/articles_resave.json'
        )
        self.dataset_reflect['radiofeatures_dataset'] = radiofeatures_dataset
        self.data_whole_3D = self.data_whole_3D + [{'radiofeatures_dataset': i} for i in range(len(radiofeatures_dataset))]
        print('radiofeatures_dataset loaded')
        
        # Combine all datasets
        self.data_whole = self.data_whole_2D + self.data_whole_3D
        
            
    def __len__(self):
        """Return the total number of samples in the combined dataset"""
        return len(self.data_whole)
    
    def __getitem__(self, idx):
        """
        Get a specific sample from the dataset
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Dictionary containing processed inputs for model training
        """
        # Get sample from the appropriate dataset
        sample = list(self.data_whole[idx].items())[0]
        dataset_index = sample[0]
        sample = self.dataset_reflect[sample[0]][sample[1]] 
        '''
        Dict: {
            "image_dict": [
                            {"image": image, # image is a tensor of shape [c,w,h,d], c is channel=3, w is width, h is height, d is depth(1 for chestxray,pmcoa,pmcvqa)
                            "position": {"question": 0}}, position is a dict, random choice of 0 or len(question)
                        ]
            "question": question, 
            "answer":answer,  
            }
        '''
        images = sample["image_dict"]
        question = sample["question"]
        answer = sample["answer"]
        
        # Process text and images together
        images, question, answer = self.text_add_image(images, question, answer)
        
        # Stack images into a single tensor
        try:
            vision_x = stack_images(images)
        except:
            print(self.data_whole[idx].items())
        
        # Tokenize combined question and answer text
        self.text_tokenizer.padding_side = "right"
        text_tensor = self.text_tokenizer(
            question + ' ' + answer, max_length=self.max_seq, truncation=True, padding="max_length", return_tensors="pt"
        )
        lang_x = text_tensor["input_ids"][0]
        attention_mask = text_tensor["attention_mask"][0]
        
        # Add EOS token if necessary
        try:
            lang_x[torch.sum(attention_mask)] = self.text_tokenizer.eos_token_id
        except:
            pass
            
        # Extract important medical terms for loss weighting
        emphasize_words = []
        emphasize_words = [str(_) for _ in self.words_extract.extract(answer)]
        
        if emphasize_words != []:
            emphasize_words_tensor = self.text_tokenizer(
                emphasize_words, max_length=self.max_seq
            )
            key_embeddings = [torch.tensor(_[1:]) for _ in emphasize_words_tensor['input_ids']]
        else:
            key_embeddings = []
            
        # Tokenize question to determine its length
        question_tensor = self.text_tokenizer(
            question, max_length=self.max_seq, truncation=True, padding="max_length", return_tensors="pt"
        )
        question_length = torch.sum(question_tensor["attention_mask"][0])
        
        # Create labels for training (ignore question tokens, padding, and special tokens)
        labels = lang_x.clone()
        labels[labels == self.text_tokenizer.pad_token_id] = -100
        labels[labels >= self.voc_size] = -100
        labels[:question_length] = -100
        
        # Apply weights to important medical terms in the labels
        reweight_tensor = find_position(labels, key_embeddings)
        
        # Don't use keyword emphasis for certain datasets
        if dataset_index == 'paper_inline_dataset':
            emphasize_words = []
        
        # Return all processed inputs for model training
        return {
            'vision_x': vision_x,
            'lang_x': lang_x, 
            'attention_mask': attention_mask, 
            'labels': labels, 
            'loss_reweight': reweight_tensor, 
            'key_words_query': emphasize_words
        }
    
    def text_add_image(self, images, question, answer):
        """
        Insert image tokens into text at appropriate positions
        
        Args:
            images: List of image tensors
            question: Question text
            answer: Answer text
            
        Returns:
            Tuple of (processed_images, question_with_image_tokens, answer_with_image_tokens)
        """
        ref_image = []
        question = str(question)
        answer = str(answer)
        
        # Initialize lists to track image insertion positions
        question_list = [[] for _ in range(len(str(question)))]
        answer_list = [[] for _ in range(len(str(answer)))]
        
        # Process each image and its position
        for index, image in enumerate(images):
            ref_image.append(image["image"])
            position = image["position"]
            position = list(position.items())[0]
            
            # Store image indices at their insertion positions
            if position[0] == 'question':
                insert_loc = position[1] - 1
                if insert_loc < 0:
                    insert_loc = 0
                question_list[insert_loc].append(index)
            if position[0] == 'answer':
                insert_loc = position[1] - 1
                if insert_loc < 0:
                    insert_loc = 0
                answer_list[insert_loc].append(index)
                
        # Build new question text with image tokens
        new_question = ''
        new_answer = ''
        question = str(question)
        for char_i in range(len(question)):
            if question_list[char_i] == []:
                new_question = new_question + question[char_i]
            if question_list[char_i] != []:
                for img_index in question_list[char_i]:
                    try:
                        new_question = new_question + '<image>' + self.image_padding_tokens[img_index] + '</image>'
                    except:
                        print("Error: out of max image input size")
                new_question = new_question + question[char_i]
                
        # Build new answer text with image tokens
        answer = str(answer)
        for char_i in range(len(str(answer))):
            if answer_list[char_i] == []:
                new_answer = new_answer + answer[char_i]
            if answer_list[char_i] != []:
                for img_index in answer_list[char_i]:
                    try:
                        new_answer = new_answer + '<image>' + self.image_padding_tokens[img_index] + '</image>'
                    except:
                        print("Error: out of max image input size")
                new_answer = new_answer + answer[char_i]
                
        # Remove bullet points for cleaner text
        new_answer = new_answer.replace('•', '')
        return ref_image, new_question, new_answer
                
        
        
        
        
        
# torch.set_printoptions(profile="full")    
# text_tokenizer = '/home/cs/leijiayu/wuchaoyi/Finetune_LLAMA/LLAMA_Model/tokenizer'
# dataset = multi_dataset(text_tokenizer = text_tokenizer)
# print(len(dataset))
# for i in range(10):
#     dataset[i]
#     input()