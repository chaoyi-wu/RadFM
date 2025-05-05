# Import necessary libraries
from torch import nn
from transformers.models.llama import LlamaForCausalLM
from .my_embedding_layer import MyEmbedding
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import tqdm.auto as tqdm
import torch.nn as nn
import torch
from torch.utils.checkpoint import checkpoint
from torch.autograd import Variable
import numpy as np

class MultiLLaMAForCausalLM(nn.Module):
    """
    A multimodal LLaMA model that combines language and vision inputs
    for causal language modeling tasks.
    """
    def __init__(self, lang_model_path):  
        """
        Initialize the multimodal model.
        
        Args:
            lang_model_path (str): Path to the pretrained language model
        """
        super(MultiLLaMAForCausalLM, self).__init__()  
        
        # Load pretrained LLaMA model
        self.lang_model = LlamaForCausalLM.from_pretrained(
            lang_model_path,
        )
        
        # Enable gradient checkpointing for memory efficiency
        self.lang_model.gradient_checkpointing_enable()
        self.lang_model.enable_input_require_grads()
        
        # Initialize custom embedding layer and share weights with language model
        self.embedding_layer = MyEmbedding()
        self.embedding_layer.weight = self.lang_model.get_input_embeddings().weight
        
        # Set model dimensions
        self.hidden_dim = 5120
        self.voc_size = 32000
        
    def forward(self, lang_x, vision_x, attention_mask, labels, loss_reweight, key_words_query):
        """
        Forward pass for the multimodal model.
        
        Args:
            lang_x: Language input tokens
            vision_x: Vision input features
            attention_mask: Attention mask for language inputs
            labels: Target labels for language modeling
            loss_reweight: Weights for calculating loss (to prioritize certain tokens)
            key_words_query: Query for highlighting important words
            
        Returns:
            Dictionary containing model outputs including loss and logits
        """
        if labels.shape == lang_x.shape:
            # Set embedding mode to handle text inputs
            self.embedding_layer.flag = 'Text'
            
            # Get embeddings and matching loss from embedding layer
            input_embedding, loss_match = self.embedding_layer(lang_x, vision_x, key_words_query)
            
            # Forward pass through the language model
            output = self.lang_model(inputs_embeds=input_embedding, attention_mask=attention_mask, labels=labels)
            logits = output['logits']

            # Initialize regularization loss
            loss_reg = None
            if labels is not None:
                # Shift logits and labels for next-token prediction
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                shift_loss_reweight = loss_reweight[..., 1:].contiguous()
                
                # Prepare for loss calculation
                loss_fct = CrossEntropyLoss(reduction='none')
                shift_logits = shift_logits.view(-1, self.voc_size)
                shift_labels = shift_labels.view(-1)
                shift_loss_reweight = shift_loss_reweight.view(-1)
                
                # Ensure tensors are on the same device
                shift_labels = shift_labels.to(shift_logits.device)
                shift_loss_reweight = shift_loss_reweight.to(shift_logits.device) 
                
                # Calculate weighted cross-entropy loss
                loss_reg = loss_fct(shift_logits, shift_labels)
                loss_reg = torch.sum(shift_loss_reweight * loss_reg) / torch.sum(shift_loss_reweight)
            
            # Combine losses
            loss = loss_reg
            if loss_match is not None:
                loss = 0.8 * loss + 0.2 * loss_match
            
            # Calculate accuracy metrics
            logits = output['logits'][..., :-1, :].contiguous().detach()
            total = len(labels)
            predictions = torch.argmax(logits, dim=-1)
            labels = labels[..., 1:].contiguous()
            
            # Count correct predictions (ignoring padding tokens with -100)
            Acc = torch.sum(torch.all(torch.logical_or(predictions == labels, labels == -100), dim=-1))       
            Accuracy = Acc / total      
            
            return dict(
                # loss_reg = loss_reg,
                # loss_matching = loss_matching,
                logits=Accuracy,
                loss=output['loss'],
            )
            
        ### useless for now ignore the folowing codes ###
        # if labels.shape == vision_x.shape:
        #    self.embedding_layer.flag = 'Seg'
        #    input_embedding = self.embedding_layer(lang_x, vision_x)
    
    def generate(self, lang_x, vision_x):
        """
        Generate text based on language and vision inputs.
        
        Args:
            lang_x: Language input tokens
            vision_x: Vision input features
            
        Returns:
            Generated token sequence
        """
        # Set embedding mode to text generation
        self.embedding_layer.flag = 'Text'
        
        with torch.no_grad():
            # Get embeddings from the embedding layer
            input_embedding, _ = self.embedding_layer(lang_x, vision_x) 
            
            # Generate text using language model
            generation = self.lang_model.generate(
                inputs_embeds=input_embedding, 
                max_new_tokens=200,
                top_k=50
            )
            
        return generation