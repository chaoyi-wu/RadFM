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


def find_position(label, key_embeddings):
    loss_reweight = torch.ones(label.shape)
    for i in range(len(label)):
        if label[i] == -100:
            loss_reweight[i] = 0
        else:
            for key_embedding in key_embeddings:
                if torch.equal(label[i:i+len(key_embedding)], key_embedding):
                    loss_reweight[i:i+len(key_embedding)] = 3
    return loss_reweight

def stack_images(images):
    
    target_H = 512
    target_W = 512
    target_D = 4
    if len(images) == 0:
        return torch.zeros((1,3,target_H,target_W,target_D))
    MAX_D = 4
    D_list = list(range(4,65,4))
    
    for ii in images:
        try:
            D = ii.shape[3]
            if D > MAX_D:
                MAX_D = D
        except:
            continue
    for temp_D in D_list:
        if abs(temp_D - MAX_D)< abs(target_D - MAX_D):
            target_D = temp_D
            
    stack_images = []
    for s in images:
        if len(s.shape) == 3:
        #print(s.shape)
            stack_images.append(torch.nn.functional.interpolate(s.unsqueeze(0).unsqueeze(-1), size = (target_H,target_W,target_D)))
        else:
            stack_images.append(torch.nn.functional.interpolate(s.unsqueeze(0), size = (target_H,target_W,target_D)))
    images = torch.cat(stack_images, dim=0)
    return images

class multi_dataset_close(Dataset):
    def __init__(self, text_tokenizer, test_split = 'close', max_seq = 2048, max_img_size = 10, image_num=32,voc_size =32000, down_sample_ratio = 100):
        
        self.down_sample_ratio = down_sample_ratio
        self.text_tokenizer = text_tokenizer
        self.max_img_size = max_img_size
        self.image_num = image_num
        self.max_seq = max_seq
        self.voc_size = voc_size
        self.H = 512
        self.W = 512
        self.image_padding_tokens = []
        if isinstance(self.text_tokenizer,str):
            self.text_tokenizer = LlamaTokenizer.from_pretrained(
                self.text_tokenizer,
            )
            special_token = {"additional_special_tokens": ["<image>","</image>"]}
            for i in range(max_img_size):
                image_padding_token = ""
                for j in range(image_num):
                    image_token = "<image"+str(i*image_num+j)+">"
                    image_padding_token = image_padding_token + image_token
                    special_token["additional_special_tokens"].append("<image"+str(i*image_num+j)+">")
                self.image_padding_tokens.append(image_padding_token)
            self.text_tokenizer.add_special_tokens(
                special_token
            )
            self.text_tokenizer.pad_token_id = 0
            self.text_tokenizer.bos_token_id = 1
            self.text_tokenizer.eos_token_id = 2


        self.data_whole_2D = []
        self.data_whole_3D = []
        self.dataset_reflect = {}
        self.test_split = test_split
        ### closed ###
        if self.test_split == 'close':

            chestxray_dataset_bn = Binary_Dataset(csv_path = '/gpfs/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/data_csv/chestxray_balance_test.csv',  
                                prompt_json_file = '/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/dataset/yes_no_prompt.json')
            self.dataset_reflect['chestxray_dataset_bn'] = chestxray_dataset_bn
            self.data_whole_2D = self.data_whole_2D +  [{'chestxray_dataset_bn':i} for i in range(len(chestxray_dataset_bn))]
            print('chestxray_dataset_bn loaded')
            
            pcxr_dataset_bn = Binary_Dataset(csv_path = '/gpfs/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/data_csv/pcxr_balance_test.csv',  
                                        prompt_json_file = '/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/dataset/yes_no_prompt.json')
            self.dataset_reflect['pcxr_dataset_bn'] = pcxr_dataset_bn
            self.data_whole_2D = self.data_whole_2D +  [{'pcxr_dataset_bn':i} for i in range(len(pcxr_dataset_bn))]
            print('pcxr_dataset_bn loaded')
            
            mammo_dataset_bn = Binary_Dataset(csv_path = '/gpfs/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/data_csv/mammo_balance_test.csv',  
                                        prompt_json_file = '/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/dataset/yes_no_prompt.json')
            self.dataset_reflect['mammo_dataset_bn'] = mammo_dataset_bn
            self.data_whole_2D = self.data_whole_2D +  [{'mammo_dataset_bn':i} for i in range(len(mammo_dataset_bn))]
            print('mammo_dataset_bn loaded')
            
            spinexr_dataset_bn = Binary_Dataset(csv_path = '/gpfs/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/data_csv/spinexr_balance_test.csv',  
                                        prompt_json_file = '/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/dataset/yes_no_prompt.json')
            self.dataset_reflect['spinexr_dataset_bn'] = spinexr_dataset_bn
            self.data_whole_2D = self.data_whole_2D +  [{'spinexr_dataset_bn':i} for i in range(len(spinexr_dataset_bn))]
            print('spinexr_dataset_bn loaded')
            
            radiomodality_dataset = Radio_Modality_Dataset(csv_path = '/gpfs/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/data_csv/radio_modality_test.csv',  
                                prompt_json_file = '/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/dataset/modality_prompt.json',
                                modality_json_file = '/gpfs/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/data_csv/modality_set.json')
            self.dataset_reflect['radiomodality_dataset'] = radiomodality_dataset
            self.data_whole_3D = self.data_whole_3D +  [{'radiomodality_dataset':i} for i in range(len(radiomodality_dataset))]
            print('radiomodality_dataset loaded')
            
            ### multi-label ###
            chestxray_dataset = ChestXray_Dataset(csv_path = '/gpfs/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/data_csv/chestxray_test.csv',  
                                prompt_json_file = '/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/dataset/cls_prompt.json')
            self.dataset_reflect['chestxray_dataset'] = chestxray_dataset
            self.data_whole_2D = self.data_whole_2D +  [{'chestxray_dataset':i} for i in range(len(chestxray_dataset))]
            print('chestxray_dataset loaded')
            
            pcxr_dataset = ChestXray_Dataset(csv_path = '/gpfs/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/data_csv/pcxr_test.csv',  
                                prompt_json_file = '/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/dataset/cls_prompt.json')
            self.dataset_reflect['pcxr_dataset'] = pcxr_dataset
            self.data_whole_2D = self.data_whole_2D +  [{'pcxr_dataset':i} for i in range(len(pcxr_dataset))]
            print('pcxr_dataset loaded')
            
            mammo_dataset = ChestXray_Dataset(csv_path = '/gpfs/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/data_csv/mammo_test.csv',  
                                        prompt_json_file = '/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/dataset/mammo_prompt.json')
            self.dataset_reflect['mammo_dataset'] = mammo_dataset
            self.data_whole_2D = self.data_whole_2D +  [{'mammo_dataset':i} for i in range(len(mammo_dataset))]
            print('mammo_dataset loaded')
            
            spinexr_dataset = ChestXray_Dataset(csv_path = '/gpfs/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/data_csv/spinexr_test.csv',  
                                        prompt_json_file = '/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/dataset/spinexr_prompt.json')
            self.dataset_reflect['spinexr_dataset'] = spinexr_dataset
            self.data_whole_2D = self.data_whole_2D +  [{'spinexr_dataset':i} for i in range(len(spinexr_dataset))]
            print('spinexr_dataset loaded')

            medpix_single_dataset = MedPix_Single_Dataset(csv_path = '/gpfs/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/data_csv/MedPix_single_test_close.csv')
            self.dataset_reflect['medpix_single_dataset'] = medpix_single_dataset
            self.data_whole_2D = self.data_whole_2D +  [{'medpix_single_dataset':i} for i in range(len(medpix_single_dataset))]
            print('medpix_single_dataset loaded')
            
            vqarad_dataset = VQA_Dataset(csv_path = '/gpfs/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/data_csv/vqarad_test.csv')
            self.dataset_reflect['vqarad_dataset'] = vqarad_dataset
            self.data_whole_2D = self.data_whole_2D +  [{'vqarad_dataset':i} for i in range(len(vqarad_dataset))]
            print('vqarad_dataset loaded')
            
            slake_dataset = VQA_Dataset(csv_path = '/gpfs/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/data_csv/slakevqa_test.csv')
            self.dataset_reflect['slake_dataset'] = slake_dataset
            self.data_whole_2D = self.data_whole_2D +  [{'slake_dataset':i} for i in range(len(slake_dataset))]
            print('slake_dataset loaded')
        
        
        self.data_whole = self.data_whole_2D + self.data_whole_3D
        random.shuffle(self.data_whole)
        
            
    def __len__(self):
        return math.ceil(len(self.data_whole)/self.down_sample_ratio)
    
    def __getitem__(self, idx):
        idx = (self.down_sample_ratio*idx +random.randint(0,self.down_sample_ratio-1))%len(self.data_whole)
        # vision_x, lang_x, attention_mask, labels
        sample = list(self.data_whole[idx].items())[0]
        dataset_index = sample[0]
        #print(sample)
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
        question = str(sample["question"])
        answer = str(sample["answer"])
        images, question, answer = self.text_add_image(images,question,answer)
        
        # print(question,answer)
        ### make vision_x
        try:
            vision_x = stack_images(images)
        except:
            print(self.data_whole[idx].items())
        #print(vision_x.shape,question,answer)
        
        ### make lang_x ###
        self.text_tokenizer.padding_side = "right"
        text_tensor = self.text_tokenizer(
            question + ' ' + answer, max_length=self.max_seq, truncation=True, padding="max_length", return_tensors="pt"
        )
        lang_x = text_tensor["input_ids"][0]
        attention_mask = text_tensor["attention_mask"][0]
        try:
            lang_x[torch.sum(attention_mask)] = self.text_tokenizer.eos_token_id
        except:
            pass
        ### make label ###
        
        emphasize_words = []
        # emphasize_words =  [str(_) for _ in self.words_extract.extract(answer)]
        
        if emphasize_words != []:
            emphasize_words_tensor  = self.text_tokenizer(
                emphasize_words , max_length=self.max_seq
            )
            key_embeddings = [torch.tensor(_[1:]) for _ in emphasize_words_tensor['input_ids']]
        else:
            key_embeddings = []
        question_tensor = self.text_tokenizer(
            question, max_length=self.max_seq, truncation=True, padding="max_length", return_tensors="pt"
        )
        question_length  = torch.sum(question_tensor["attention_mask"][0])
        labels = lang_x.clone()
        labels[labels == self.text_tokenizer.pad_token_id] = -100
        labels[labels >= self.voc_size] = -100
        labels[:question_length] = -100
        
        reweight_tensor = find_position(labels, key_embeddings)
        if dataset_index == 'paper_inline_dataset':
            emphasize_words = []
        # print(labels,key_embeddings,reweight_tensor)
        return {'vision_x': vision_x,'lang_x':lang_x, 'attention_mask': attention_mask, 'labels':labels, 'loss_reweight': reweight_tensor, 'key_words_query': emphasize_words}
    
    def text_add_image(self,images,question,answer):
        question = str(question)
        answer = str(answer)
        ref_image = []
        question_list = [[] for _ in range(len(str(question)))]
        answer_list = [[] for _ in range(len(str(answer)))]
        for index, image in enumerate(images):
            ref_image.append(image["image"])
            position = image["position"]
            position = list(position.items())[0]
            if position[0] == 'question':
                insert_loc = position[1] -1
                if insert_loc < 0:
                    insert_loc = 0
                question_list[insert_loc].append(index)
            if position[0] == 'answer':
                insert_loc = position[1] -1
                if insert_loc < 0:
                    insert_loc = 0
                answer_list[insert_loc].append(index)
        new_question = ''
        new_answer = ''
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
        
        for char_i in range(len(answer)):
            if answer_list[char_i] == []:
                new_answer = new_answer + answer[char_i]
            if answer_list[char_i] != []:
                for img_index in answer_list[char_i]:
                    try:
                        new_answer = new_answer + '<image>' + self.image_padding_tokens[img_index] + '</image>'
                    except:
                        print("Error: out of max image input size")
                new_answer = new_answer + answer[char_i]
                
        new_answer = new_answer.replace('•','')
        return ref_image,new_question,new_answer
                
        
        
        
        
        
# torch.set_printoptions(profile="full")    
# text_tokenizer = '/home/cs/leijiayu/wuchaoyi/Finetune_LLAMA/LLAMA_Model/tokenizer'
# dataset = multi_dataset_close(text_tokenizer = text_tokenizer)
# print(len(dataset))
# for i in range(10):
#     dataset[i]
#     input()
