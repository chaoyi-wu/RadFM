# Import necessary libraries for data processing, model loading, and inference
import tqdm.auto as tqdm
import torch.nn.functional as F
from typing import Optional, Dict, Sequence
from typing import List, Optional, Tuple, Union
import transformers
from dataclasses import dataclass, field
from Model.RadFM.multimodality_model import MultiLLaMAForCausalLM
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from torchvision import transforms
from PIL import Image   

def get_tokenizer(tokenizer_path, max_img_size=100, image_num=32):
    '''
    Initialize the tokenizer with special tokens for image handling
    
    Args:
        tokenizer_path: Path to the base tokenizer
        max_img_size: Maximum number of images supported in a prompt
        image_num: Number of token embeddings per image
        
    Returns:
        Tuple of (tokenizer, image_padding_tokens)
    '''
    if isinstance(tokenizer_path, str):
        image_padding_tokens = []
        # Load the base tokenizer from the provided path
        text_tokenizer = LlamaTokenizer.from_pretrained(
            tokenizer_path,
        )
        # Define initial special tokens for image markup
        special_token = {"additional_special_tokens": ["<image>", "</image>"]}
        
        # Generate unique tokens for each image position and patch
        for i in range(max_img_size):
            image_padding_token = ""
            
            for j in range(image_num):
                image_token = "<image" + str(i * image_num + j) + ">"
                image_padding_token = image_padding_token + image_token
                special_token["additional_special_tokens"].append("<image" + str(i * image_num + j) + ">")
            
            # Store the concatenated tokens for each image
            image_padding_tokens.append(image_padding_token)
            
            # Add all special tokens to the tokenizer
            text_tokenizer.add_special_tokens(
                special_token
            )
            
            # Configure standard special tokens for LLaMA models
            text_tokenizer.pad_token_id = 0
            text_tokenizer.bos_token_id = 1
            text_tokenizer.eos_token_id = 2    
    
    return text_tokenizer, image_padding_tokens    

def combine_and_preprocess(question, image_list, image_padding_tokens):
    '''
    Combine text and images into a multimodal input format
    
    Args:
        question: Text input or question to process
        image_list: List of images with their metadata
        image_padding_tokens: Special tokens for image placeholders
        
    Returns:
        Tuple of (processed_text, processed_images_tensor)
    '''
    # Define image transformation pipeline
    transform = transforms.Compose([                        
                transforms.RandomResizedCrop([512, 512], scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
            ])
    
    images = []
    new_qestions = [_ for _ in question]  # Convert question string to list of characters
    padding_index = 0
    
    # Process each image in the list
    for img in image_list:
        img_path = img['img_path']
        position = img['position']  # Where to insert the image in the text
        
        # Load and transform the image
        image = Image.open(img_path).convert('RGB')   
        image = transform(image)
        image = image.unsqueeze(0).unsqueeze(-1)  # Add batch and depth dimensions (c,w,h,d)
        
        # Resize the image to target dimensions
        target_H = 512 
        target_W = 512 
        target_D = 4 
        # This can be different for 3D and 2D images. For demonstration we here set this as the default sizes for 2D images. 
        images.append(torch.nn.functional.interpolate(image, size=(target_H, target_W, target_D)))
        
        # Insert image placeholder token at the specified position in text
        new_qestions[position] = "<image>" + image_padding_tokens[padding_index] + "</image>" + new_qestions[position]
        padding_index += 1
    
    # Stack all images into a batch and add batch dimension
    vision_x = torch.cat(images, dim=1).unsqueeze(0)  # Cat tensors and expand the batch_size dim
    
    # Join the character list back into a string
    text = ''.join(new_qestions) 
    return text, vision_x
    
    
def main():
    '''
    Main function to demonstrate the RadFM model inference
    '''
    print("Setup tokenizer")
    # Initialize tokenizer with special image tokens
    text_tokenizer, image_padding_tokens = get_tokenizer('./Language_files')
    print("Finish loading tokenizer")
    
    ### Initialize a simple case for demo ###
    print("Setup demo case")
    # Define a medical question about a chest X-ray
    question = "Can you identify any visible signs of Cardiomegaly in the image?"
    
    # Specify the image path and where to insert it in the question
    image = [
            {
                'img_path': './view1_frontal.jpg',
                'position': 0,  # Insert at the beginning of the question
            },  # Can add arbitrary number of images
        ] 
    
    # Combine text and images into model-ready format
    text, vision_x = combine_and_preprocess(question, image, image_padding_tokens)    
    print("Finish loading demo case")
    
    print("Setup Model")
    # Initialize the multimodal model
    model = MultiLLaMAForCausalLM(
        lang_model_path='./Language_files',  # Build up model based on LLaMa-13B config
    )
    
    # Load pretrained model weights
    ckpt = torch.load('./pytorch_model.bin', map_location='cpu')  # Please download our checkpoint from huggingface and decompress the original zip file first
    model.load_state_dict(ckpt)
    print("Finish loading model")
    
    # Move model to GPU and set to evaluation mode
    model = model.to('cuda')
    model.eval() 
    
    # Run inference without gradient computation
    with torch.no_grad():
        # Tokenize the combined text with image placeholders
        lang_x = text_tokenizer(
                text, max_length=2048, truncation=True, return_tensors="pt"
        )['input_ids'].to('cuda')
        
        # Move image tensor to GPU
        vision_x = vision_x.to('cuda')
        
        # Generate text response
        generation = model.generate(lang_x, vision_x)
        
        # Decode the generated token IDs to text
        generated_texts = text_tokenizer.batch_decode(generation, skip_special_tokens=True) 
        
        # Print results
        print('---------------------------------------------------')
        print('Input: ', question)
        print('Output: ', generated_texts[0])

    
if __name__ == "__main__":
    main()