import tqdm.auto as tqdm
import torch.nn.functional as F
from typing import Optional, Dict, Sequence
from typing import List, Optional, Tuple, Union
import transformers
from My_Trainer.trainer import Trainer
from dataclasses import dataclass, field
from Dataset.multi_dataset_test import multi_dataset
from Model.RadFM.multimodality_model import MultiLLaMAForCausalLM
from datasampler import My_DistributedBatchSampler
import torch
from torch.utils.data import DataLoader  
import csv
import random
import numpy as np
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(20)
# 预处理数据以及训练模型

@dataclass
class ModelArguments:
    lang_encoder_path: Optional[str] = field(default="/home/cs/leijiayu/wuchaoyi/book_pretrain/Results/Book_mix_2048_13B_full/checkpoint-45800")
    tokenizer_path: str = field(default='/home/cs/leijiayu/wuchaoyi/Finetune_LLAMA/LLAMA_Model/tokenizer', metadata={"help": "Path to the tokenizer data."})   
    #vision_encoder_path: str = field(default='/home/cs/leijiayu/wuchaoyi/multi_modal/src/PMC-CLIP/checkpoint.pt', metadata={"help": "Path to the vision_encoder."})   
    

@dataclass
class DataArguments:
    Mode: Optional[str] = field(default="Train")
    test_split: Optional[str] = field(default="open")
    
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    remove_unused_columns: bool = field(default = False)
    batch_size_2D: int = field(default = 4)
    batch_size_3D: int = field(default = 1)
    output_dir: Optional[str] = field(default="/home/cs/leijiayu/wuchaoyi/multi_modal/src/Results/BLIP_overfit/")
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")


@dataclass
class DataCollator(object):

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        #print(instances)
        vision_xs, lang_xs, attention_masks, labels = tuple([instance[key] for instance in instances] for key in ('vision_x','lang_x', 'attention_mask', 'labels'))
        
        lang_xs = torch.cat([_.unsqueeze(0) for _ in lang_xs],dim  = 0)
        attention_masks = torch.cat([_.unsqueeze(0) for _ in attention_masks],dim  = 0)
        labels = torch.cat([_.unsqueeze(0) for _ in labels],dim  = 0)
        #print(lang_xs.shape,attention_masks.shape,labels.shape)
        target_H = 512
        target_W = 512
        target_D = 4
        MAX_D = 0
        
        if len(vision_xs) == 1:
            target_H = 256
            target_W = 256
        
           
        D_list = list(range(4,65,4))
        if len(vision_xs) == 1:
            if vision_xs[0].shape[0] >6:
                D_list = list(range(4,33,4))
    
        for ii in vision_xs:
            try:
                D = ii.shape[-1]
                if D > MAX_D:
                    MAX_D = D
            except:
                continue
        for temp_D in D_list:
            if abs(temp_D - MAX_D)< abs(target_D - MAX_D):
                target_D = temp_D

        vision_xs = [torch.nn.functional.interpolate(s, size = (target_H,target_W,target_D)) for s in vision_xs]
        
        vision_xs = torch.nn.utils.rnn.pad_sequence(
            vision_xs, batch_first=True, padding_value=0
        )
        print(vision_xs.shape)
        return dict(
            lang_x=lang_xs,
            vision_x=vision_xs,
            attention_mask=attention_masks,
            labels = labels,
        )
                 
def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    training_args.data_sampler = My_DistributedBatchSampler
    print("Setup Data")
    Test_dataset = multi_dataset(text_tokenizer = model_args.tokenizer_path,test_split = data_args.test_split)
    
    Test_dataloader = DataLoader(
            Test_dataset,
            batch_size=1,
            num_workers=1,
            pin_memory=True,
            sampler=None,
            shuffle=True,
            collate_fn=None,
            drop_last=False,
    )  
    
    print("Setup Model")
    model = MultiLLaMAForCausalLM(
        lang_model_path=model_args.lang_encoder_path,
    )
    ckpt = torch.load('/gpfs/home/cs/leijiayu/wuchaoyi/wangyingjie/src/Results/backup/checkpoint-17600/pytorch_model.bin',map_location ='cpu')
    # ckpt.pop('embedding_layer.figure_token_weight')
    model.load_state_dict(ckpt,strict=False)
    model = model.to('cuda')
    model.eval() 
    with open('output_whole_2_epoch' + data_args.test_split+'.csv', mode='w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["Question", "Ground Truth","Pred",'belong_to'])
        cc = 0
        for sample in tqdm.tqdm(Test_dataloader):
            
            question = sample["question"]
            belong_to = sample['belong_to']
            # img_pp = sample['img_path']
            lang_x = Test_dataset.text_tokenizer(
                question, max_length=2048, truncation=True, return_tensors="pt"
            )['input_ids'].to('cuda')
            vision_x = sample["vision_x"].to('cuda')
            answer = sample['answer']
            try:
                generation = model.generate(lang_x,vision_x)
                generated_texts = Test_dataset.text_tokenizer.batch_decode(generation, skip_special_tokens=True) 
                writer.writerow([question,answer,generated_texts,belong_to])
                cc = cc+1
            # if cc>=10000:
            #     break
            except:
                continue

if __name__ == "__main__":
    main()
       