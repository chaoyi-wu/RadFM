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
    def __init__(self, csv_path,img_path):
        self.img_path = img_path
        self.question_list = pd.read_csv(csv_path)
        
        #normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.transform = transforms.Compose([                        
                transforms.RandomResizedCrop([512,512],scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                #normalize,
            ])   
                
        
    def __len__(self):
        return len(self.question_list)
    
    def __getitem__(self, idx):
        # lang_x, vision_x, attention_mask, labels
        sample = self.question_list.iloc[idx]
        
        PMC_id = sample['PMC_id']
        img_ref = literal_eval(sample['img_ref'])
        context = str(sample['context'])
        sentences = context.split('.')
        if len(sentences) > 5:
            first_sentence = sentences[0]
            last_sentences = ". ".join(context.split('.')[-4:])
            context = first_sentence + '. ' + last_sentences
        question = str(context) + '\n' + str(sample['question']).replace('Q:','') 
        answer = str(sample['answer']).replace('A:','')
        
        images = []
        for img_id in img_ref:
            #print(img_ref)
            img_path = self.img_path + '/' + PMC_id+'_'+ img_id + '.jpg'
            try:
                image = Image.open(img_path).convert('RGB')   
                image = self.transform(image)
                
                p = random.random()
                if random.random() >0.5:
                    images.append({'image':image, "position": {"question":len(question)}})    
                else:
                    images.append({'image':image, "position": {"question":len(context)}}) 
            except:
                continue        
    
        return {
            "image_dict": images,
            "question": question,
            "answer":answer,
            }

# csv_path = '/gpfs/home/cs/leijiayu/wuchaoyi/multi_modal/Data/GPT_realdata/casa_report_train.csv'    
# img_path = '/home/cs/leijiayu/data/all_images/figures/'
# dataset = CaseReport_dataset(csv_path, img_path)
# print(dataset[0])
