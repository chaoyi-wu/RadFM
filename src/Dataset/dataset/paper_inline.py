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
    def __init__(self, csv_path,img_path, sample_sentence_length = 50,max_img_size = 3):
        self.max_img_size = max_img_size
        self.sample_sentence_length = sample_sentence_length
        self.img_path = img_path
        self.paper_path = np.array(pd.read_csv(csv_path)['PMC_path'])
        
        #normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.transform = transforms.Compose([                        
                transforms.RandomResizedCrop([512,512],scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                #normalize,
            ])   
                
        
    def __len__(self):
        return self.paper_path.shape[0]
    
    def __getitem__(self, idx):
        # lang_x, vision_x, attention_mask, labels
        paper_json = self.paper_path[idx]
        PMC_name = paper_json.rsplit('/',2)[-1].split('.')[0]
        sentences_list = json.load(open(paper_json,'r'))
        image_dict, question, answer = self.random_sample_sentence(sentences_list,PMC_name)
        return {
            "image_dict": image_dict,
            "question": question,
            "answer":answer,
            }

    def random_sample_sentence(self, sentences_list, PMC_name):
        sentences_length = len(sentences_list)
        p = random.random()
        if p>=0.5:
            if len(sentences_list) > self.sample_sentence_length:
                start = random.randint(0, sentences_length-self.sample_sentence_length)
                sentences_list = sentences_list[start:(start + self.sample_sentence_length)]
        else:
            if len(sentences_list) > self.sample_sentence_length:
                sample_start = []
                for sentence_id in range(len(sentences_list)):
                    if sentences_list[sentence_id]['img_ref'] != []:
                        if sentence_id-10 < 0:
                            sample_start.append(0)
                        else:
                            if sentence_id-10 > sentences_length-self.sample_sentence_length:
                                sample_start.append(sentences_length-self.sample_sentence_length)
                            else:
                                sample_start.append(sentence_id-10)
                if sample_start == []:
                    start = random.randint(0, sentences_length-self.sample_sentence_length)
                    sentences_list = sentences_list[start:(start + self.sample_sentence_length)]
                else:
                    start = sample_start[random.randint(0, len(sample_start)-1)]
                    sentences_list = sentences_list[start:(start + self.sample_sentence_length)]
            
        text = ''
        images = []
        for ix in sentences_list:
            sentence = ix
            if sentence["img_ref"] == []:
                text = text + sentence['text']
            else:
                if len(images)+len(sentence["img_ref"]) > self.max_img_size:
                    break
                for img_id in sentence["img_ref"]:
                    img_path = self.img_path + '/' + PMC_name+'_'+ img_id + '.jpg'
                    if os.path.exists(img_path):
                        try:
                            image = Image.open(img_path).convert('RGB')   
                            image = self.transform(image)
                            images.append({'image':image, "position": {"answer":len(text)}})    
                        except:
                            continue
                text = text + sentence['text']            
        question = ''
        answer = text
        
        return images,question,answer

# csv_path = '/home/cs/leijiayu/wuchaoyi/multi_modal/Data/train_paper.csv'    
# img_path = '/home/cs/leijiayu/data/all_images/figures/'
# dataset = multi_paper_dataset(csv_path, img_path)
# print(dataset[0])
