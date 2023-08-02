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
        s = torch.tensor(s)
        if len(s.shape) == 3:
        #print(s.shape)
            stack_images.append(torch.nn.functional.interpolate(s.unsqueeze(0).unsqueeze(-1), size = (target_H,target_W,target_D)))
        else:
            stack_images.append(torch.nn.functional.interpolate(s.unsqueeze(0), size = (target_H,target_W,target_D)))
    images = torch.cat(stack_images, dim=0)
    return images

class multi_dataset(Dataset):
    def __init__(self, text_tokenizer, test_split = 'close', max_seq = 2048, max_img_size = 10, image_num=32,voc_size =32000):
        
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
        if self.test_split == 'diagnosis':
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
            
        if self.test_split == 'modality':
            radiomodality_dataset = Radio_Modality_Dataset(csv_path = '/gpfs/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/data_csv/radio_modality_test.csv',  
                                prompt_json_file = '/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/dataset/modality_prompt.json',
                                modality_json_file = '/gpfs/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/data_csv/modality_set.json',down_sample_ratio = 1)
            self.dataset_reflect['radiomodality_dataset'] = radiomodality_dataset
            self.data_whole_3D = self.data_whole_3D +  [{'radiomodality_dataset':i} for i in range(len(radiomodality_dataset))]
            print('radiomodality_dataset loaded')
            
            # medpix_single_dataset = MedPix_Single_Dataset(csv_path = '/gpfs/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/data_csv/MedPix_single_test_close.csv')
            # self.dataset_reflect['medpix_single_dataset'] = medpix_single_dataset
            # self.data_whole_2D = self.data_whole_2D +  [{'medpix_single_dataset':i} for i in range(len(medpix_single_dataset))]
            # print('medpix_single_dataset loaded')
            
        if self.test_split == 'vqa':
            
            # medpix_qa_dataset = MedPix_QA_Dataset(csv_path = '/gpfs/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/data_csv/MedPix_questions_test.csv')
            # self.dataset_reflect['medpix_qa_dataset'] = medpix_qa_dataset
            # self.data_whole_2D = self.data_whole_2D +  [{'medpix_qa_dataset':i} for i in range(len(medpix_qa_dataset))]
            # print('medpix_qa_dataset loaded')
            
            pmcvqa_dataset = PMCVQA_Dataset(csv_path = '/gpfs/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/data_csv/pmcvqa_test.csv')
            self.dataset_reflect['pmcvqa_dataset'] = pmcvqa_dataset
            self.data_whole_2D = self.data_whole_2D +  [{'pmcvqa_dataset':i} for i in range(len(pmcvqa_dataset))]
            print('pmcvqa_dataset loaded')
            
            casereport_dataset = CaseReport_dataset(csv_path = '/gpfs/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/data_csv/filtered_case_report_test.csv',    
                                    img_path = '/home/cs/leijiayu/data/all_images/figures/')
            self.dataset_reflect['casereport_dataset'] = casereport_dataset
            self.data_whole_2D = self.data_whole_2D +  [{'casereport_dataset':i} for i in range(len(casereport_dataset))]
            print('casereport_dataset loaded')
            
            vqarad_dataset = PMCVQA_Dataset(csv_path = '/gpfs/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/data_csv/vqarad_test.csv')
            self.dataset_reflect['vqarad_dataset'] = vqarad_dataset
            self.data_whole_2D = self.data_whole_2D +  [{'vqarad_dataset':i} for i in range(len(vqarad_dataset))]
            print('vqarad_dataset loaded')
            
            slake_dataset = PMCVQA_Dataset(csv_path = '/gpfs/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/data_csv/slakevqa_test.csv')
            self.dataset_reflect['slake_dataset'] = slake_dataset
            self.data_whole_2D = self.data_whole_2D +  [{'slake_dataset':i} for i in range(len(slake_dataset))]
            print('slake_dataset loaded')
            
            ## 3D
            radiovqa_dataset = RadioVQA_Dataset(csv_path = '/gpfs/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/data_csv/radiology_vqa_test.csv')
            self.dataset_reflect['radiovqa_dataset'] = radiovqa_dataset
            self.data_whole_3D = self.data_whole_3D +  [{'radiovqa_dataset':i} for i in range(len(radiovqa_dataset))]
            print('radiovqa_dataset loaded')
        
        if self.test_split == 'caption':
            ## open ###
            # medpix_multi_dataset = MedPix_Multi_Dataset(csv_path = '/gpfs/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/data_csv/MedPix_multi_test.csv')
            # self.dataset_reflect['medpix_multi_dataset'] = medpix_multi_dataset
            # self.data_whole_2D = self.data_whole_2D +  [{'medpix_multi_dataset':i} for i in range(len(medpix_multi_dataset))]
            # print('medpix_multi_dataset loaded')
            
            chestxray_caption_dataset = ChestXray_Dataset(csv_path = '/gpfs/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/data_csv/mimic_caption_test.csv',  
                                prompt_json_file = '/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/dataset/report_prompt.json')
            self.dataset_reflect['chestxray_caption_dataset'] = chestxray_caption_dataset
            self.data_whole_2D = self.data_whole_2D +  [{'chestxray_caption_dataset':i} for i in range(len(chestxray_caption_dataset))]
            print('chestxray_caption_dataset loaded')
            
            # medpix_single_dataset = MedPix_Single_Dataset(csv_path = '/gpfs/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/data_csv/MedPix_single_test.csv')
            # self.dataset_reflect['medpix_single_dataset'] = medpix_single_dataset
            # self.data_whole_2D = self.data_whole_2D +  [{'medpix_single_dataset':i} for i in range(len(medpix_single_dataset))]
            # print('medpix_single_dataset loaded')

            radiocaption_dataset = RadioCaption_Dataset(json_path='/gpfs/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/data_csv/radiology_article_npy_test.json',
                                                        prompt_json_file = '/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/dataset/caption_prompt.json'
                                                        )
            self.dataset_reflect['radiocaption_dataset'] = radiocaption_dataset
            self.data_whole_3D = self.data_whole_3D +  [{'radiocaption_dataset':i} for i in range(len(radiocaption_dataset))]
            print('radiocaption_dataset loaded')
            
        if self.test_split == 'feature':    
            radiofeatures_dataset = Radiofeatures_Dataset(json_path='/gpfs/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/data_csv/radiology_article_npy_test.json',
                                                    prompt_json_file = '/gpfs/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/dataset/radiology_feature_prompt.json',
                                                    disease_prompt_json_file = '/gpfs/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/dataset/yes_no_prompt.json',
                                                    article_json_file = '/gpfs/home/cs/leijiayu/wuchaoyi/wangyingjie/src/New_Dataset/data_csv/articles_resave.json')
            self.dataset_reflect['radiofeatures_dataset'] = radiofeatures_dataset
            self.data_whole_3D = self.data_whole_3D +  [{'radiofeatures_dataset':i} for i in range(len(radiofeatures_dataset))]
            print('radiofeatures_dataset loaded')
        
        
        self.data_whole = self.data_whole_2D + self.data_whole_3D
        
            
    def __len__(self):
        return len(self.data_whole)
    
    def __getitem__(self, idx):
        # vision_x, lang_x, attention_mask, labels
        sample = list(self.data_whole[idx].items())[0]
        #print(sample)
        belong_to = sample[0]
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
        if len(images) > 8:
            images = random.sample(images,8)
        question = str(sample["question"])
        answer = str(sample["answer"])
        images, question, answer = self.text_add_image(images,question,answer)
        
        #print(question,answer)
        ### make vision_x
        try:
            vision_x = stack_images(images)
        except:
            print(self.data_whole[idx].items())
            input()
        #print(vision_x.shape,question,answer)

        return {'vision_x': vision_x,'question':question, 'answer':answer, 'belong_to':belong_to,}
    
    def text_add_image(self,images,question,answer):
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
# dataset = multi_dataset(text_tokenizer = text_tokenizer)
# print(len(dataset))
# for i in range(10):
#     dataset[i]
#     input()