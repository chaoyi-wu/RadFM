# Import necessary libraries
import torch.nn as nn
import torch.nn.functional as F
import torch 
from .helpers import PerceiverResampler    
from .utils import get_visual_encoder
from einops import rearrange, repeat
from einops_exts import rearrange_many
import torchvision
from .vit_3d import ViT
from einops.layers.torch import Rearrange
from .transformer_decoder import TransformerDecoder, TransformerDecoderLayer
from torch.utils.checkpoint import checkpoint
from torch.autograd import Variable
import random
from transformers import AutoTokenizer, AutoModel

class MyEmbedding(nn.Module):
    """
    Custom embedding layer for multimodal inputs that combines text and vision features.
    """
    def __init__(self, num_embeddings=32000, embedding_dim=5120, perceiver_num=32, vis_dim=768, 
                 patch_size=32, frame_patch_size=4, seg_channel=256):
        """
        Initialize the multimodal embedding layer.
        
        Args:
            num_embeddings (int): Size of vocabulary for text embeddings
            embedding_dim (int): Dimension of output embeddings
            perceiver_num (int): Number of latent vectors in perceiver
            vis_dim (int): Dimension of vision features
            patch_size (int): Size of image patches
            frame_patch_size (int): Size of 3D frame patches
            seg_channel (int): Number of segmentation channels
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Standard embedding weight matrix for text tokens
        self.weight = nn.Parameter(torch.torch.randn((num_embeddings, embedding_dim)))
        # Special token weights for figures/images
        self.figure_token_weight = nn.Parameter(torch.randn((2, embedding_dim)))
        self.flag = 'Text'  # Mode flag: 'Text' or 'Seg'
        self.patch_size = patch_size 
        self.frame_patch_size = frame_patch_size
        self.seg_channel = seg_channel
        
        ## the MedKEBERT can be downloaded from https://huggingface.co/xmcmic/Med-KEBERT/tree/main ##
        # Initialize medical domain BERT model for keyword understanding
        self.bert_tokenizer = AutoTokenizer.from_pretrained("xmcmic/Med-KEBERT")
        self.bert_model = AutoModel.from_pretrained("xmcmic/Med-KEBERT")
        # Project BERT outputs to vision feature space
        self.bert_projection_fc = nn.Linear(768, vis_dim)
        
        # 3D Vision Transformer for processing volumetric medical images
        self.vision_encoder = ViT(
            image_size=512,           # image size
            frames=512,               # max number of frames
            image_patch_size=patch_size,     # image patch size
            frame_patch_size=frame_patch_size,      # frame patch size
            dim=vis_dim,
            depth=12,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )
        
        # Upscaling layers for vision features (used in segmentation mode)
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose3d(vis_dim, vis_dim // 4, kernel_size=2, stride=2),
            nn.BatchNorm3d(vis_dim // 4),
            nn.GELU(),
            nn.ConvTranspose3d(vis_dim // 4, vis_dim // 8, kernel_size=2, stride=2),
            nn.GELU(),
        )
        
        # Transformer decoder for cross-attention between text and vision
        decoder_layer = TransformerDecoderLayer(d_model=vis_dim, nhead=8, normalize_before=True)
        decoder_norm = nn.LayerNorm(vis_dim)
        self.transformer_decoder = TransformerDecoder(decoder_layer=decoder_layer, num_layers=4, norm=decoder_norm)
        
        # MLP for processing transformer decoder outputs
        self.transformer_decoder_mlp = nn.Sequential(
            nn.Linear(vis_dim, vis_dim // 4),
            nn.GELU(),
            nn.Linear(vis_dim // 4, vis_dim // 8),
            nn.GELU(),
        )
        self.vis_dim = vis_dim
        
        # Perceiver resampler to reduce sequence length of vision features
        self.perceiver = PerceiverResampler(dim=self.vis_dim, num_latents=perceiver_num)
        # Final projection to embedding dimension
        self.fc = nn.Linear(self.vis_dim, self.embedding_dim)
        # Classification head for matching keywords
        self.cls_head = nn.Linear(self.vis_dim // 8, 1)
        

    def forward(self, text_input, vision_x, key_words_query=None):
        """
        Forward pass for the embedding layer.
        
        Args:
            text_input: Text token indices [B, L]
            vision_x: Visual input features [B, S, C, H, W, D]
            key_words_query: Optional list of key words for contrastive learning
            
        Returns:
            tuple: (output_embeddings, loss_matching)
                - output_embeddings: Combined embeddings for text and vision
                - loss_matching: Contrastive loss for keyword matching (or None)
        """
        if self.flag == 'Text':
            # Process in text mode
            B, S, C, H, W, D = vision_x.shape
            # Reshape for batch processing
            vision_x = rearrange(vision_x, "b S c h w d-> (b S) c h w d")
            
            # Process through vision encoder
            vision_x, pos_embedding = self.vision_encoder(vision_x)
            
            # Reshape back to batch format
            vision_x = rearrange(vision_x, "(b s F) v d -> b s F v d", b=B, s=S, F=1) 
            
            loss_matching = None
             
            if key_words_query is not None:
                ## we do not use the following parts in final version. 
                ## You can quota the following codes and if so the bert models will be useless.
                # key_words_query list[list[str]] B, words, each word matches corresponding vision_x embedding
                
                # Extract and deduplicate keywords
                query_words = [item for sublist in key_words_query for item in sublist]
                query_words = list(set(query_words))
                
                # Limit number of keywords to process
                if len(query_words) > 16:
                    random.shuffle(query_words)
                    query_words = query_words[0:16]
                    
                if query_words != []:
                    # Create binary labels for contrastive learning
                    contrastive_labels = torch.zeros(B, len(query_words))  # B Q
                    for i, sublist in enumerate(key_words_query):
                        for j, item in enumerate(query_words):
                            if item in sublist:
                                contrastive_labels[i, j] = 1 
                    contrastive_labels = contrastive_labels.to(vision_x.dtype).to(vision_x.device)        
                    
                    # Get BERT embeddings for keywords
                    with torch.no_grad():
                        query_words_embedding = self.bert_tokenizer(
                            query_words, 
                            padding='max_length', 
                            truncation=True, 
                            max_length=256,
                            return_tensors="pt"
                        )
                        query_words_embedding = self.bert_model(
                            input_ids=query_words_embedding['input_ids'].to(vision_x.device),
                            attention_mask=query_words_embedding['attention_mask'].to(vision_x.device)
                        )['last_hidden_state'][:, 0, :].to(vision_x.dtype).to(vision_x.device)  # Q,D
                        
                    # Project BERT embeddings to vision space
                    query_words_embedding = self.bert_projection_fc(query_words_embedding)
                    query_words_embedding = query_words_embedding.unsqueeze(0).repeat(B, 1, 1)  # B,Q,D
                    _, N, _ = query_words_embedding.shape
                    
                    # Pool vision features
                    image_embedding = vision_x.mean(dim=1)  # B V D average pooling to remove multimodality
                    image_embedding = rearrange(image_embedding, "b F v d -> b (F v) d")
                    pos_embedding = rearrange(pos_embedding, "(b s) v d -> b s v d", b=B, s=S)[:, 0, :, :]
                    
                    # Prepare inputs for transformer decoder
                    image_embedding = image_embedding.transpose(0, 1)  # (H/P W/P D/P) B D
                    pos_embedding = pos_embedding.transpose(0, 1)  # (H/P W/P D/P) B D
                    query_words_embedding = query_words_embedding.transpose(0, 1)  # N B D
                    
                    # Cross-attention between keywords and image features
                    oo_embedding, _ = self.transformer_decoder(
                        query_words_embedding, image_embedding, pos=pos_embedding
                    ) 
                    oo_embedding = oo_embedding.transpose(0, 1)  # B Q D
                    oo_embedding = rearrange(oo_embedding, 'b n d -> (b n) d')
                    oo_embedding = self.transformer_decoder_mlp(oo_embedding)
                    oo_embedding = self.cls_head(oo_embedding).mean(dim=-1)
                    oo_embedding = rearrange(oo_embedding, '(b n) -> b n', b=B, n=N)  # B Q 
                    
                    # Calculate contrastive loss
                    loss_matching = F.binary_cross_entropy_with_logits(oo_embedding, contrastive_labels) 
                
            # Process vision features through perceiver resampler
            vision_x = self.perceiver(vision_x)  # reshapes to (b, S, n, d)
            
            n = vision_x.shape[2]
            
            # Project vision features to embedding dimension
            vision_x = rearrange(vision_x, "b s n d -> (b s n) d")
            vision_x = self.fc(vision_x)
            vision_x = rearrange(vision_x, "(b T) d -> b T d", b=B, T=n*S)
            
            # Combine text and vision embeddings
            embedding_weight = torch.cat([self.weight, self.figure_token_weight], dim=0)
            embedding_weight = embedding_weight.unsqueeze(0).repeat(B, 1, 1)
            embedding_weight = torch.cat([embedding_weight, vision_x], dim=1)
            
            # Convert text indices to one-hot and compute final embeddings
            text_input = F.one_hot(text_input, embedding_weight.shape[1]).to(vision_x.dtype).to(vision_x.device)
            out_put = torch.matmul(text_input, embedding_weight)
            
        ## useless for now. ignore the folowing code##    
        # if self.flag == 'Seg':
        #    B,C,H,W,D =  vision_x.shape
        #    _,N,_ = text_input.shape
        #    latent_embedding, pos_embedding = self.vision_encoder(vision_x) # B (H/P W/P D/P) D
            
        #    image_embedding = latent_embedding.transpose(0,1) # (H/P W/P D/P) B  D
        #    pos_embedding = pos_embedding.transpose(0,1) # (H/P W/P D/P) B  D
        #    text_input = text_input.transpose(0,1) # N B D
            
        #    mask_embedding,_ = self.transformer_decoder(text_input, image_embedding, pos = pos_embedding) 
        #    mask_embedding = mask_embedding.transpose(0,1) # B N D
        #    mask_embedding = rearrange(mask_embedding, 'b n d -> (b n) d')
        #    mask_embedding = self.transformer_decoder_mlp(mask_embedding)
        #    mask_embedding = rearrange(mask_embedding, '(b n) d -> b n d', b=B, n=N,d = self.vis_dim // 8)
            
        #    vision_x = rearrange(latent_embedding,'b (h w d) c -> b c h w d', h = (H // self.patch_size), w = (W // self.patch_size), d = (D // self.frame_patch_size), c=self.vis_dim)
        #    vision_x = self.output_upscaling(vision_x) #B C H/4 W/4 D/4
        #    out_put = torch.einsum('bchwd,bnc->bnhwd', vision_x, mask_embedding)
        
        return out_put, loss_matching

# model = MyEmbedding(vision_encoder_path = '')
# text_input = torch.randint(low=0, high=3210, size=(4,2048))
# image_input = torch.randn((4,3,3,512,512,4))
# key_words_query = [[],[],[],['consoliation']]
# print(model(text_input, image_input, key_words_query))