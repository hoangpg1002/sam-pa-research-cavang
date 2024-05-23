# Copyright by HQ-SAM team
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
import random
from typing import Dict, List, Tuple
from segment_anything_training import sam_model_registry
from segment_anything_training.modeling import TwoWayTransformer, MaskDecoder,ImageEncoderViT
from utils.dataloader import get_im_gt_name_dict, create_dataloaders, RandomHFlip, Resize, LargeScaleJitter
from utils.loss_mask import loss_masks
import utils.misc as misc
import sys

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Type
from efficientnet_pytorch import EfficientNet

from segment_anything_training.modeling.common import LayerNorm2d, MLPBlock
def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(42)
class Downsample(nn.Module):
    def __init__(self, in_channels):
        super(Downsample, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.conv(x)

# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(ConvBlock, self).__init__()
#         self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=3//2, groups=in_channels)
#         self.layer_norm = LayerNorm2d(in_channels)
#         self.pointwise_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#         self.gelu = nn.GELU()
#         self.pointwise_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
#         self.drop_path = nn.Dropout2d(p=0.1)  
#         self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
#     def forward(self, x):
#         residual = x
#         residual = self.residual_conv(residual)
#         x = self.depthwise_conv(x)
#         x = self.layer_norm(x)
#         x = self.pointwise_conv1(x)
#         x = self.gelu(x)
#         x = self.drop_path(x)
#         x += residual
#         return x
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=3//2, groups=in_channels)
        self.layer_norm = LayerNorm2d(in_channels)
        self.pointwise_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.gelu = nn.GELU()
        self.pointwise_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.layer_scale = nn.Parameter(torch.ones(out_channels, 1, 1))
        self.drop_path = nn.Dropout2d(p=0.1)  
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
    def forward(self, x):
        residual = x
        residual = self.residual_conv(residual)
        x = self.depthwise_conv(x)
        x = self.layer_norm(x)
        x = self.pointwise_conv1(x)
        x = self.gelu(x)
        x = self.pointwise_conv2(x)
        x = self.layer_scale * x
        x = self.drop_path(x)
        x += residual
        return x
class ConvNextEncoder(nn.Module):
    def __init__(self, in_channels, block_channels):
        super(ConvNextEncoder, self).__init__()
        blocks = []
        num_blocks = len(block_channels)
        for i in range(num_blocks):
            if i > 0:
                blocks.append(Downsample(block_channels[i-1]))
            blocks.append(ConvBlock(in_channels if i == 0 else block_channels[i-1], block_channels[i]))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)

class FeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, block_channels=[64, 128, 256, 512, 768]):
        super(FeatureExtractor, self).__init__()
        self.encoder = ConvNextEncoder(in_channels, block_channels)

    def forward(self, x):
        return self.encoder(x)

class CrossBranchAdapter(nn.Module):
    def __init__(self):
        super(CrossBranchAdapter, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels=1536,out_channels=1536,kernel_size=7, padding=3, stride=1,groups=1536),nn.Sigmoid())
        #self.upchannel=nn.Conv2d(in_channels=512,out_channels=768,kernel_size=1,stride=1)
        self.downchannel=nn.Conv2d(in_channels=1536,out_channels=768,kernel_size=1,stride=1)
        self.max_pool = nn.AdaptiveMaxPool2d((64,64))
        self.mean_pool = nn.AdaptiveAvgPool2d((64,64))
    def forward(self, tensor1, tensor2):
        # Concatenate 2 tensors along the channel dimension
        concat_tensor = tensor1.permute(0, 3, 1, 2) + tensor2 #([1, 768, 64, 64])
        shortcut=concat_tensor

        # Max and Mean pooling operations on concat_tensor

        max_pooled = self.max_pool(concat_tensor) #torch.Size([1, 768, 64, 64])
        mean_pool = self.mean_pool(concat_tensor)
        #max_pooled=self.HW(max_pooled)
        #mean_pool=self.HW(mean_pool)
        pooled_concat=torch.cat([max_pooled,mean_pool],dim=1)
        conv_out=self.conv(pooled_concat)
        conv_out=self.downchannel(conv_out)
        # Convolutional layer
        conv_out = conv_out * shortcut + shortcut#torch.Size([1, 768, 64, 64])
        #conv_out = self.mlp(conv_out.permute(0,2,3,1))
        #print(conv_out.shape) #torch.Size([1, 768, 64, 64])
        #conv_out=self.mlp(conv_out.permute(0,2,3,1)) 
        return conv_out.permute(0,2,3,1)
# class MLPBlock(nn.Module):
#     def __init__(
#         self,
#         embedding_dim: int,
#         mlp_dim: int,
#         out_dim:int,
#         act: Type[nn.Module] = nn.GELU,
#     ) -> None:
#         super().__init__()
#         self.lin1 = nn.Linear(embedding_dim, mlp_dim)
#         self.lin2 = nn.Linear(mlp_dim, out_dim)
#         self.act = act()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.lin2(self.act(self.lin1(x)))   
# class CrossBranchAdapter(nn.Module):
#     def __init__(self):
#         super(CrossBranchAdapter, self).__init__()
#         self.mean_pool = nn.AdaptiveMaxPool2d((64,64))
#         self.mlp_block_1=MLPBlock(embedding_dim=512,mlp_dim=512*4,out_dim=256,act=nn.GELU)
#         self.sigmoid = nn.Sigmoid()
#         # self.h1 = nn.Linear(4096, 64)
#         # self.h2 = nn.Linear(64, 4096)
#     def forward(self, tensor1, tensor2):
#         # Concatenate 2 tensors along the channel dimension
#         Fc=tensor1
#         Ft=tensor2
#         concat_tensor = torch.cat([tensor1,tensor2],dim=1) #(1,512,64,64)

#         # Max and Mean pooling operations on concat_tensor
#         mean_pool = self.mean_pool(concat_tensor) #(1,512,64,64)
#         Wc=self.sigmoid(self.mlp_block_1(mean_pool.permute(0,2,3,1))).permute(0,3,1,2)
#         Wt=self.sigmoid(self.mlp_block_1(mean_pool.permute(0,2,3,1))).permute(0,3,1,2)
#         Filterc=torch.mul(Fc,Wc)
#         Filtert=torch.mul(Ft,Wt)
#         RecC=Filterc+Fc
#         RecT=Filtert+Ft
#         Ac = torch.exp(RecC) / (torch.exp(RecC) + torch.exp(RecT))
#         At = torch.exp(RecT) / (torch.exp(RecT) + torch.exp(RecC))
#         final_feature=Ac*Fc+At*Ft
#         return final_feature
# class CrossBranchAdapter(nn.Module):
#     def __init__(self):
#         super(CrossBranchAdapter, self).__init__()
#         self.max_pool = nn.AdaptiveMaxPool2d((64,64))
#         self.mean_pool = nn.AdaptiveAvgPool2d((64,64))
#         #self.mlp_block_2=MLPBlock(embedding_dim=512,mlp_dim=512*2,out_dim=256,act=nn.GELU)
#         self.conv = nn.Sequential(nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3, padding=1, stride=1),nn.Sigmoid())
#         self.dchannels = nn.Sequential(nn.Conv2d(in_channels=512,out_channels=256,kernel_size=1, stride=1),LayerNorm2d(256),nn.GELU())
#         self.sigmoid = nn.Sigmoid()
#         # self.h1 = nn.Linear(4096, 64)
#         # self.h2 = nn.Linear(64, 4096)
#     def forward(self, tensor1, tensor2):
#         # Concatenate 2 tensors along the channel dimension
#         concat_tensor = tensor1+tensor2 #(1,256,64,64)
#         shortcut_concat= concat_tensor
#         # Max and Mean pooling operations on concat_tensor
#         mean_pooled = self.mean_pool(concat_tensor) #(1,256,64,64)
#         max_pooled = self.max_pool(concat_tensor)
#         pooled_concat=torch.cat([max_pooled,mean_pooled],dim=1)
#         w_conv=self.conv(pooled_concat)
#         w_conv=self.dchannels(w_conv)
#         final_feature=shortcut_concat*w_conv+shortcut_concat
#         return final_feature
# This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
class DualImageEncoderViT(ImageEncoderViT):
    def __init__(self,model_type,is_train):
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__(
            img_size = 1024,
            patch_size = 16,
            in_chans = 3,
            embed_dim = 768,
            depth = 12,
            num_heads = 12,
            mlp_ratio = 4.0,
            out_chans= 256,
            qkv_bias = True,
            norm_layer = nn.LayerNorm,
            act_layer = nn.GELU,
            use_abs_pos = True,
            use_rel_pos = True,
            rel_pos_zero_init = True,
            window_size = 14,
            global_attn_indexes = [2, 5, 8, 11],
        )
        assert model_type in ["vit_b","vit_l","vit_h"]
        
        checkpoint_dict = {"vit_b":"/kaggle/working/training/pretrained_checkpoint/sam_vit_b_imageencoder.pth",
                           "vit_l":"pretrained_checkpoint/sam_vit_l_maskdecoder.pth",
                           'vit_h':"pretrained_checkpoint/sam_vit_h_maskdecoder.pth"}
        checkpoint_path = checkpoint_dict[model_type]
        self.load_state_dict(torch.load(checkpoint_path),strict=False)
        print("Dual Image Encoder init from SAM ImageEncoder")
        for name, param in self.named_parameters():
            if 'cross_branch_adapter' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        self.feature_extractor=FeatureExtractor()
        self.cross_branch_adapter=CrossBranchAdapter()
        if is_train==True:
            self.load_state_dict(torch.load("/kaggle/working/training/pretrained_checkpoint/epoch_5encoder.pth"))
            print("encoder load pretrained!")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
            add_features=self.feature_extractor(x)
            x = self.patch_embed(x) #(1,64,64,768)
            if self.pos_embed is not None:
                x = x + self.pos_embed

            interm_embeddings=[]
            for blk in self.blocks:
                x = blk(x,add_features)
                if blk.window_size == 0:
                    interm_embeddings.append(x)
            x=self.cross_branch_adapter(x,add_features)
            x = self.neck(x.permute(0, 3, 1, 2))
            return x, interm_embeddings



#==============================================

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x

class MaskDecoderHQ(MaskDecoder):
    def __init__(self, model_type,is_train):
        super().__init__(transformer_dim=256,
                        transformer=TwoWayTransformer(
                                depth=2,
                                embedding_dim=256,
                                mlp_dim=2048,
                                num_heads=8,
                            ),
                        num_multimask_outputs=3,
                        activation=nn.GELU,
                        iou_head_depth= 3,
                        iou_head_hidden_dim= 256,)
        assert model_type in ["vit_b","vit_l","vit_h"]
        
        checkpoint_dict = {"vit_b":"/kaggle/working/training/pretrained_checkpoint/sam_vit_b_maskdecoder.pth",
                           "vit_l":"pretrained_checkpoint/sam_vit_l_maskdecoder.pth",
                           'vit_h':"pretrained_checkpoint/sam_vit_h_maskdecoder.pth"}
        checkpoint_path = checkpoint_dict[model_type]
        self.load_state_dict(torch.load(checkpoint_path))
        print("HQ Decoder init from SAM MaskDecoder")
        for n,p in self.named_parameters():
            p.requires_grad = False
        transformer_dim=256
        vit_dim_dict = {"vit_b":768,"vit_l":1024,"vit_h":1280}
        vit_dim = vit_dim_dict[model_type]

        self.hf_token = nn.Embedding(1, transformer_dim)
        self.hf_mlp = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        self.num_mask_tokens = self.num_mask_tokens + 1

        self.compress_vit_feat = nn.Sequential(
                                            nn.ConvTranspose2d(vit_dim, transformer_dim, kernel_size=2, stride=2),
                                            LayerNorm2d(transformer_dim),
                                            nn.GELU(), 
                                            nn.ConvTranspose2d(transformer_dim, transformer_dim // 8, kernel_size=2, stride=2))
            
        self.embedding_encoder = nn.Sequential(
                                        nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
                                        LayerNorm2d(transformer_dim // 4),
                                        nn.GELU(),
                                        nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
                                    )

        self.embedding_maskfeature = nn.Sequential(
                                        nn.Conv2d(transformer_dim // 8, transformer_dim // 4, 3, 1, 1), 
                                        LayerNorm2d(transformer_dim // 4),
                                        nn.GELU(),
                                        nn.Conv2d(transformer_dim // 4, transformer_dim // 8, 3, 1, 1))
        if is_train==True:
            self.load_state_dict(torch.load("/kaggle/working/training/pretrained_checkpoint/epoch_5decoder.pth"))
            print("decoder load pretrained!")

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        hq_token_only: bool,
        interm_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the ViT image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted hq masks
        """

        vit_features = interm_embeddings[0].permute(0, 3, 1, 2) #interm_embeddings[0] =(1,64,64,768) => (1,768,64,64)
        hq_features=self.embedding_encoder(image_embeddings)+self.compress_vit_feat(vit_features)
        batch_len = len(image_embeddings)
        masks = []
        iou_preds = []
        for i_batch in range(batch_len):
            mask, iou_pred = self.predict_masks(
                image_embeddings=image_embeddings[i_batch].unsqueeze(0),
                image_pe=image_pe[i_batch],
                sparse_prompt_embeddings=sparse_prompt_embeddings[i_batch],
                dense_prompt_embeddings=dense_prompt_embeddings[i_batch],
                hq_feature = hq_features[i_batch].unsqueeze(0)
            )
            masks.append(mask)
            iou_preds.append(iou_pred)
        masks = torch.cat(masks,0)
        iou_preds = torch.cat(iou_preds,0)

        # Select the correct mask or masks for output
        if multimask_output:
            # mask with highest score
            mask_slice = slice(1,self.num_mask_tokens-1)
            iou_preds = iou_preds[:, mask_slice]
            iou_preds, max_iou_idx = torch.max(iou_preds,dim=1)
            iou_preds = iou_preds.unsqueeze(1)
            masks_multi = masks[:, mask_slice, :, :]
            masks_sam = masks_multi[torch.arange(masks_multi.size(0)),max_iou_idx].unsqueeze(1)
        else:
            # singale mask output, default
            mask_slice = slice(0, 1)
            masks_sam = masks[:,mask_slice]

        masks_hq = masks[:,slice(self.num_mask_tokens-1, self.num_mask_tokens), :, :]
        if hq_token_only:
            return masks_hq
        else:
            return masks_sam, masks_hq

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        hq_feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""

        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight, self.hf_token.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0) 
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)

        upscaled_embedding_sam = self.output_upscaling(src)
        upscaled_embedding_ours = self.embedding_maskfeature(upscaled_embedding_sam) + hq_feature
        
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            if i < 4:
                hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
            else:
                hyper_in_list.append(self.hf_mlp(mask_tokens_out[:, i, :]))

        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding_sam.shape

        masks_sam = (hyper_in[:,:4] @ upscaled_embedding_sam.view(b, c, h * w)).view(b, -1, h, w)
        masks_ours = (hyper_in[:,4:] @ upscaled_embedding_ours.view(b, c, h * w)).view(b, -1, h, w)
        masks = torch.cat([masks_sam,masks_ours],dim=1)
        
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred



def show_anns(masks, input_point, input_box, input_label, filename, image, ious, boundary_ious):
    if len(masks) == 0:
        return

    for i, (mask, iou, biou) in enumerate(zip(masks, ious, boundary_ious)):
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        if input_box is not None:
            show_box(input_box, plt.gca())
        if (input_point is not None) and (input_label is not None): 
            show_points(input_point, input_label, plt.gca())

        plt.axis('off')
        plt.savefig(filename+'_'+str(i)+'.png',bbox_inches='tight',pad_inches=-0.1)
        plt.close()

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    


# def get_args_parser():
#     parser = argparse.ArgumentParser('HQ-SAM', add_help=False)

#     parser.add_argument("--output", type=str, required=True, default="/train/",
#                         help="Path to the directory where masks and checkpoints will be output")
#     parser.add_argument("--model-type", type=str, default="vit_b", 
#                         help="The type of model to load, in ['vit_h', 'vit_l', 'vit_b']")
#     parser.add_argument("--checkpoint", type=str, required=True, default="train\pretrained_checkpoint",
#                         help="The path to the SAM checkpoint to use for mask generation.")
#     parser.add_argument("--device", type=str, default="cuda", 
#                         help="The device to run generation on.")

#     parser.add_argument('--seed', default=42, type=int)
#     parser.add_argument('--learning_rate', default=1e-3, type=float)
#     parser.add_argument('--start_epoch', default=0, type=int)
#     parser.add_argument('--lr_drop_epoch', default=10, type=int)
#     parser.add_argument('--max_epoch_num', default=12, type=int)
#     parser.add_argument('--input_size', default=[1024,1024], type=list)
#     parser.add_argument('--batch_size_train', default=4, type=int)
#     parser.add_argument('--batch_size_valid', default=1, type=int)
#     parser.add_argument('--model_save_fre', default=1, type=int)

#     # parser.add_argument('--world_size', default=1, type=int,
#     #                     help='number of distributed processes')
#     parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
#     # parser.add_argument('--rank', default=0, type=int,
#     #                     help='number of distributed processes')
#     # parser.add_argument('--local_rank', type=int, help='local rank for dist')
#     parser.add_argument('--find_unused_params', action='store_true')

#     parser.add_argument('--eval', action='store_true')
#     parser.add_argument('--visualize', action='store_true')
#     parser.add_argument("--restore-model", type=str,
#                         help="The path to the hq_decoder training checkpoint for evaluation")

#     return parser.parse_args()


def main(net,encoder,train_datasets, valid_datasets):

    # misc.init_distributed_mode(args)
    # print('world size: {}'.format(args.world_size))
    # print('rank: {}'.format(args.rank))
    # print('local_rank: {}'.format(args.local_rank))
    # print("args: " + str(args) + '\n')

    # seed = args.seed + misc.get_rank()
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)

    ### --- Step 1: Train or Valid dataset ---
    print("--- create training dataloader ---")
    train_im_gt_list = get_im_gt_name_dict(train_datasets, flag="train")
    train_dataloaders, train_datasets = create_dataloaders(train_im_gt_list,
                                                    my_transforms = [
                                                                RandomHFlip(),
                                                                LargeScaleJitter()
                                                                ],
                                                    batch_size = 2,
                                                    training = True)
    print(len(train_dataloaders), " train dataloaders created")

    print("--- create valid dataloader ---")
    valid_im_gt_list = get_im_gt_name_dict(valid_datasets, flag="valid")
    valid_dataloaders, valid_datasets = create_dataloaders(valid_im_gt_list,
                                                          my_transforms = [
                                                                        Resize([1024,1024])
                                                                    ],
                                                          batch_size=1,
                                                          training=False)
    print(len(valid_dataloaders), " valid dataloaders created")
    
    ### --- Step 2: DistributedDataParallel---
    # if torch.cuda.is_available():
    #     net.cuda()
    # net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
    # net_without_ddp = net.module

 
    ### --- Step 3: Train or Evaluate ---
    print("--- define optimizer ---")
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10)
    lr_scheduler.last_epoch = 0
    train(net, encoder,optimizer, train_dataloaders, valid_dataloaders, lr_scheduler)
    sam = sam_model_registry["vit_b"](checkpoint="/kaggle/working/training/pretrained_checkpoint/sam_vit_b_01ec64.pth").to(device="cuda")
    evaluate(net,encoder,sam, valid_dataloaders)


def train(net,encoder,optimizer, train_dataloaders, valid_dataloaders, lr_scheduler):
    if misc.is_main_process():
        os.makedirs("train", exist_ok=True)

    epoch_start = 0
    epoch_num = 20
    train_num = len(train_dataloaders)

    net.train()
    _ = net.to(device="cuda")
    encoder.train()
    _ = encoder.to(device="cuda")
    print("requires_grad of imageencoder")
    for n,p in encoder.named_parameters():
        if p.requires_grad:
            print(n)
    print("requires_grad of maskdecoder")
    for n,p in net.named_parameters():
        if p.requires_grad:
            print(n)
    sam = sam_model_registry["vit_b"](checkpoint="/kaggle/working/training/pretrained_checkpoint/sam_vit_b_01ec64.pth")
    _ = sam.to(device="cuda")
    # sam = torch.nn.parallel.DistributedDataParallel(sam, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
    
    for epoch in range(epoch_start,epoch_num): 
        print("epoch:   ",epoch, "  learning rate:  ", optimizer.param_groups[0]["lr"])
        metric_logger = misc.MetricLogger(delimiter="  ")
        # train_dataloaders.batch_sampler.sampler.set_epoch(epoch)

        for data in metric_logger.log_every(train_dataloaders,1000):
            inputs, labels = data['image'], data['label']
            if torch.cuda.is_available():
                inputs = inputs.to(device="cuda")
                labels = labels.to(device="cuda")

            imgs = inputs.permute(0, 2, 3, 1).cpu().numpy()
            
            # input prompt
            input_keys = ['box','point','noise_mask']
            labels_box = misc.masks_to_boxes(labels[:,0,:,:])
            try:
                labels_points = misc.masks_sample_points(labels[:,0,:,:])
            except:
                # less than 10 points
                input_keys = ['box','noise_mask']
            labels_256 = F.interpolate(labels, size=(256, 256), mode='bilinear')
            labels_noisemask = misc.masks_noise(labels_256)

            batched_input = []
            for b_i in range(len(imgs)):
                dict_input = dict()
                input_image = torch.as_tensor(imgs[b_i].astype(dtype=np.uint8), device=sam.device).permute(2, 0, 1).contiguous()
                dict_input['image'] = input_image 
                input_type = random.choice(input_keys)
                if input_type == 'box':
                    dict_input['boxes'] = labels_box[b_i:b_i+1]
                elif input_type == 'point':
                    point_coords = labels_points[b_i:b_i+1]
                    dict_input['point_coords'] = point_coords
                    dict_input['point_labels'] = torch.ones(point_coords.shape[1], device=point_coords.device)[None,:]
                elif input_type == 'noise_mask':
                    dict_input['mask_inputs'] = labels_noisemask[b_i:b_i+1]
                else:
                    raise NotImplementedError
                dict_input['original_size'] = imgs[b_i].shape[:2]
                batched_input.append(dict_input)
            # with torch.no_grad():
            #     batched_output, interm_embeddings = sam(batched_input, multimask_output=False)
            input_images = torch.stack([sam.preprocess(x=i["image"]) for i in batched_input], dim=0)
            image_embeddings, interm_embeddings = encoder(input_images)
            batched_output = []
            for image_record, curr_embedding in zip(batched_input, image_embeddings):
                if "point_coords" in image_record:
                    points = (image_record["point_coords"], image_record["point_labels"])
                else:
                    points = None
                with torch.no_grad():
                    sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                        points=points,
                        boxes=image_record.get("boxes", None),
                        masks=image_record.get("mask_inputs", None),
                    )
                    low_res_masks, iou_predictions = sam.mask_decoder(
                        image_embeddings=curr_embedding.unsqueeze(0),
                        image_pe=sam.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False
                    )
                
                masks = sam.postprocess_masks(
                    low_res_masks,
                    input_size=image_record["image"].shape[-2:],
                    original_size=image_record["original_size"],
                )
                masks = masks > sam.mask_threshold

                batched_output.append(
                    {
                        "masks": masks,
                        "iou_predictions": iou_predictions,
                        "low_res_logits": low_res_masks,
                        "encoder_embedding": curr_embedding.unsqueeze(0),
                        "image_pe": sam.prompt_encoder.get_dense_pe(),
                        "sparse_embeddings":sparse_embeddings,
                        "dense_embeddings":dense_embeddings,
                    }
                )
            batch_len = len(batched_output)
            encoder_embedding = torch.cat([batched_output[i_l]['encoder_embedding'] for i_l in range(batch_len)], dim=0)
            image_pe = [batched_output[i_l]['image_pe'] for i_l in range(batch_len)]
            sparse_embeddings = [batched_output[i_l]['sparse_embeddings'] for i_l in range(batch_len)]
            dense_embeddings = [batched_output[i_l]['dense_embeddings'] for i_l in range(batch_len)]

            masks_hq = net(
                image_embeddings=encoder_embedding,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                hq_token_only=True,
                interm_embeddings=interm_embeddings,
            )

            loss_mask, loss_dice = loss_masks(masks_hq, labels/255.0, len(masks_hq))
            loss = loss_mask + loss_dice
            
            loss_dict = {"loss_mask": loss_mask, "loss_dice":loss_dice}

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = misc.reduce_dict(loss_dict)
            losses_reduced_scaled = sum(loss_dict_reduced.values())
            loss_value = losses_reduced_scaled.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metric_logger.update(training_loss=loss_value, **loss_dict_reduced)


        print("Finished epoch:      ", epoch)
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}

        lr_scheduler.step()
        test_stats = evaluate(net,encoder,sam, valid_dataloaders)
        train_stats.update(test_stats)
        
        net.train()  
        encoder.train()

        if epoch % 1 == 0:
            model_name = "/epoch_"+str(epoch)+"decoder.pth"
            print('come here save at', "train" + model_name)
            misc.save_on_master(net.state_dict(),"train" + model_name)

            model_name = "/epoch_"+str(epoch)+"encoder.pth"
            print('come here save at', "train" + model_name)
            misc.save_on_master(encoder.state_dict(),"train" + model_name)
    
    # Finish training
    print("Training Reaches The Maximum Epoch Number")
    
    # merge sam and hq_decoder
    # if misc.is_main_process():
    #     sam_ckpt = torch.load("train\pretrained_checkpoint\sam_vit_b_01ec64.pth")
    #     hq_decoder = torch.load("train" + model_name)
    #     for key in hq_decoder.keys():
    #         sam_key = 'mask_decoder.'+key
    #         if sam_key not in sam_ckpt.keys():
    #             sam_ckpt[sam_key] = hq_decoder[key]
    #     model_name = "/sam_hq_epoch_"+str(epoch)+".pth"
    #     torch.save(sam_ckpt, "train" + model_name)



def compute_iou(preds, target):
    assert target.shape[1] == 1, 'only support one mask per image now'
    if(preds.shape[2]!=target.shape[2] or preds.shape[3]!=target.shape[3]):
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds
    iou = 0
    for i in range(0,len(preds)):
        iou = iou + misc.mask_iou(postprocess_preds[i],target[i])
    return iou / len(preds)

def compute_boundary_iou(preds, target):
    assert target.shape[1] == 1, 'only support one mask per image now'
    if(preds.shape[2]!=target.shape[2] or preds.shape[3]!=target.shape[3]):
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds
    iou = 0
    for i in range(0,len(preds)):
        iou = iou + misc.boundary_iou(target[i],postprocess_preds[i])
    return iou / len(preds)

def evaluate(net,encoder, sam, valid_dataloaders):
    net.eval()
    net.to(device="cuda")
    encoder.eval()
    encoder.to(device="cuda")
    print("Validating...")
    test_stats = {}

    for k in range(len(valid_dataloaders)):
        metric_logger = misc.MetricLogger(delimiter="  ")
        valid_dataloader = valid_dataloaders[k]
        print('valid_dataloader len:', len(valid_dataloader))

        for data_val in metric_logger.log_every(valid_dataloader,1000):
            imidx_val, inputs_val, labels_val, shapes_val, labels_ori = data_val['imidx'], data_val['image'], data_val['label'], data_val['shape'], data_val['ori_label']

            if torch.cuda.is_available():
                inputs_val = inputs_val.cuda()
                labels_val = labels_val.cuda()
                labels_ori = labels_ori.cuda()

            imgs = inputs_val.permute(0, 2, 3, 1).cpu().numpy()
            
            labels_box = misc.masks_to_boxes(labels_val[:,0,:,:])
            input_keys = ['box']
            batched_input = []
            for b_i in range(len(imgs)):
                dict_input = dict()
                input_image = torch.as_tensor(imgs[b_i].astype(dtype=np.uint8), device=sam.device).permute(2, 0, 1).contiguous()
                dict_input['image'] = input_image 
                input_type = random.choice(input_keys)
                if input_type == 'box':
                    dict_input['boxes'] = labels_box[b_i:b_i+1]
                elif input_type == 'point':
                    point_coords = labels_points[b_i:b_i+1]
                    dict_input['point_coords'] = point_coords
                    dict_input['point_labels'] = torch.ones(point_coords.shape[1], device=point_coords.device)[None,:]
                elif input_type == 'noise_mask':
                    dict_input['mask_inputs'] = labels_noisemask[b_i:b_i+1]
                else:
                    raise NotImplementedError
                dict_input['original_size'] = imgs[b_i].shape[:2]
                batched_input.append(dict_input)
            input_images = torch.stack([sam.preprocess(x=i["image"]) for i in batched_input], dim=0)
            image_embeddings, interm_embeddings = encoder(input_images)
            batched_output = []
            for image_record, curr_embedding in zip(batched_input, image_embeddings):
                if "point_coords" in image_record:
                    points = (image_record["point_coords"], image_record["point_labels"])
                else:
                    points = None
                with torch.no_grad():
                    sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                        points=points,
                        boxes=image_record.get("boxes", None),
                        masks=image_record.get("mask_inputs", None),
                    )
                    low_res_masks, iou_predictions = sam.mask_decoder(
                        image_embeddings=curr_embedding.unsqueeze(0),
                        image_pe=sam.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False
                    )
                
                masks = sam.postprocess_masks(
                    low_res_masks,
                    input_size=image_record["image"].shape[-2:],
                    original_size=image_record["original_size"],
                )
                masks = masks > sam.mask_threshold

                batched_output.append(
                    {
                        "masks": masks,
                        "iou_predictions": iou_predictions,
                        "low_res_logits": low_res_masks,
                        "encoder_embedding": curr_embedding.unsqueeze(0),
                        "image_pe": sam.prompt_encoder.get_dense_pe(),
                        "sparse_embeddings":sparse_embeddings,
                        "dense_embeddings":dense_embeddings,
                    }
                )
            batch_len = len(batched_output)
            encoder_embedding = torch.cat([batched_output[i_l]['encoder_embedding'] for i_l in range(batch_len)], dim=0)
            image_pe = [batched_output[i_l]['image_pe'] for i_l in range(batch_len)]
            sparse_embeddings = [batched_output[i_l]['sparse_embeddings'] for i_l in range(batch_len)]
            dense_embeddings = [batched_output[i_l]['dense_embeddings'] for i_l in range(batch_len)]
            
            masks_sam, masks_hq = net(
                image_embeddings=encoder_embedding,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                hq_token_only=False,
                interm_embeddings=interm_embeddings,
            )

            iou = compute_iou(masks_hq,labels_ori)
            boundary_iou = compute_boundary_iou(masks_hq,labels_ori)
                       
            loss_dict = {"val_iou_"+str(k): iou, "val_boundary_iou_"+str(k): boundary_iou}
            loss_dict_reduced = misc.reduce_dict(loss_dict)
            metric_logger.update(**loss_dict_reduced)


        print('============================')
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
        test_stats.update(resstat)
    return test_stats

if __name__ == "__main__":

    ### --------------- Configuring the Train and Valid datasets ---------------

    dataset_dis = {"name": "DIS5K-TR",
                 "im_dir": "/kaggle/input/hq44kseg/DIS5K/DIS5K/DIS-TR/im",
                 "gt_dir": "/kaggle/input/hq44kseg/DIS5K/DIS5K/DIS-TR/gt",
                 "im_ext": ".jpg",
                 "gt_ext": ".png"}

    dataset_thin = {"name": "ThinObject5k-TR",
                 "im_dir": "/kaggle/input/hq44kseg/thin_object_detection/ThinObject5K/images_train",
                 "gt_dir": "/kaggle/input/hq44kseg/thin_object_detection/ThinObject5K/masks_train",
                 "im_ext": ".jpg",
                 "gt_ext": ".png"}

    dataset_fss = {"name": "FSS",
                 "im_dir": "/kaggle/input/hq44kseg/cascade_psp/fss_all",
                 "gt_dir": "/kaggle/input/hq44kseg/cascade_psp/fss_all",
                 "im_ext": ".jpg",
                 "gt_ext": ".png"}

    dataset_duts = {"name": "DUTS-TR",
                 "im_dir": "/kaggle/input/hq44kseg/cascade_psp/DUTS-TR",
                 "gt_dir": "/kaggle/input/hq44kseg/cascade_psp/DUTS-TR",
                 "im_ext": ".jpg",
                 "gt_ext": ".png"}

    dataset_duts_te = {"name": "DUTS-TE",
                 "im_dir": "/kaggle/input/hq44kseg/cascade_psp/DUTS-TE",
                 "gt_dir": "/kaggle/input/hq44kseg/cascade_psp/DUTS-TE",
                 "im_ext": ".jpg",
                 "gt_ext": ".png"}

    dataset_ecssd = {"name": "ECSSD",
                 "im_dir": "/kaggle/input/hq44kseg/cascade_psp/ecssd",
                 "gt_dir": "/kaggle/input/hq44kseg/cascade_psp/ecssd",
                 "im_ext": ".jpg",
                 "gt_ext": ".png"}

    dataset_msra = {"name": "MSRA10K",
                 "im_dir": "/kaggle/input/hq44kseg/cascade_psp/MSRA_10K",
                 "gt_dir": "/kaggle/input/hq44kseg/cascade_psp/MSRA_10K",
                 "im_ext": ".jpg",
                 "gt_ext": ".png"}

    # valid set
    dataset_coift_val = {"name": "COIFT",
                 "im_dir": "/kaggle/input/hq44kseg/thin_object_detection/COIFT/images",
                 "gt_dir": "/kaggle/input/hq44kseg/thin_object_detection/COIFT/masks",
                 "im_ext": ".jpg",
                 "gt_ext": ".png"}

    dataset_hrsod_val = {"name": "HRSOD",
                 "im_dir": "/kaggle/input/thinobject5k/thin_object_detection/HRSOD/images",
                 "gt_dir": "/kaggle/input/thinobject5k/thin_object_detection/HRSOD/masks_max255",
                 "im_ext": ".jpg",
                 "gt_ext": ".png"}

    dataset_thin_val = {"name": "ThinObject5k-TE",
                 "im_dir": "/kaggle/input/hq44kseg/thin_object_detection/ThinObject5K/images_test",
                 "gt_dir": "/kaggle/input/hq44kseg/thin_object_detection/ThinObject5K/masks_test",
                 "im_ext": ".jpg",
                 "gt_ext": ".png"}

    dataset_dis_val = {"name": "DIS5K-VD",
                 "im_dir": "/kaggle/input/hq44kseg/DIS5K/DIS5K/DIS-VD/im",
                 "gt_dir": "/kaggle/input/hq44kseg/DIS5K/DIS5K/DIS-VD/gt",
                 "im_ext": ".jpg",
                 "gt_ext": ".png"}

    #train_datasets = [dataset_dis, dataset_thin, dataset_fss, dataset_duts, dataset_duts_te, dataset_ecssd, dataset_msra]
    train_datasets = [dataset_thin]
    valid_datasets = [dataset_thin_val] 
    #valid_datasets = [dataset_thin_val,dataset_coift_val,dataset_hrsod_val] 

    # args = get_args_parser()
    net = MaskDecoderHQ("vit_b",is_train=False) 
    encoder=DualImageEncoderViT("vit_b",is_train=False)
    main(net,encoder,train_datasets, valid_datasets)
