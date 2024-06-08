import os
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Type

from .transformer import Attention
from .common import LayerNorm2d

import utils.misc as misc

class PromptAdapater(nn.Module):
    """
    An prompt adapter layer that can adjust prompt tokens. B means predict the same image for B times with different prompts. B is 1 in the training phase.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        box_head_hidden_dim: int = 256,
        attention_downsample_rate: int = 2,
    ) -> None:
        super().__init__()
        self.static_uncertain_token = nn.Embedding(1, embedding_dim)
        self.static_refined_token = nn.Embedding(1, embedding_dim)
        self.uncertain_mlp = MLP(embedding_dim*2, embedding_dim // 4, embedding_dim, 3)
        self.refined_mlp = MLP(embedding_dim*2, embedding_dim // 4, embedding_dim, 3)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )


    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor, early_embeddings: Tensor, final_embeddings: Tensor, prompt_encoder, guiding_embedding, v_coarse, n_sample_points=4
    ) -> Tuple[Tensor, Tensor]:

        # guiding embedding generation
        b, c, h, w = guiding_embedding.shape
        input_image_size = prompt_encoder.input_image_size
        ori_h, ori_w = input_image_size
        guiding_embedding = guiding_embedding.flatten(2).permute(0, 2, 1)
        attn_out = guiding_embedding * keys
        guiding_embedding = attn_out

        # token to image cross attention
        attn_out, v_refined = self.cross_attn_token_to_image.forward_return_v(q=queries+query_pe, k=guiding_embedding+key_pe, v=guiding_embedding)
        queries = attn_out

        # generating uncertain token and refined token
        uncertain_token = torch.cat([self.static_uncertain_token.weight.repeat(b,1), queries[:,1,:]], dim=-1)
        uncertain_token = self.uncertain_mlp(uncertain_token).unsqueeze(1)
        refined_token = torch.cat([self.static_refined_token.weight.repeat(b,1), queries[:,1,:]], dim=-1)
        refined_token = self.refined_mlp(refined_token).unsqueeze(1)

        # obtain mask by U*R+(1-U)*M  (return uncertain map and mask for loss calculation)
        output_uncertain_token = uncertain_token
        output_refined_token = refined_token
        image_embedding = guiding_embedding.transpose(1, 2)
        unceratinty_map = (output_uncertain_token @ image_embedding).view(b, 1, 64, 64)
        unceratinty_map_norm = torch.sigmoid(unceratinty_map)


        refined_mask = (output_refined_token @ image_embedding).view(b, 1, 64, 64)
        coarse_mask = (queries[:,1,:] @ image_embedding).view(b, 1, 64, 64)
        final_mask = (unceratinty_map_norm>=0.5).detach()*refined_mask + (unceratinty_map_norm<0.5).detach()*coarse_mask
        mask = {"unceratinty_map": unceratinty_map_norm, 
                      "refined_mask": refined_mask,
                      "coarse_mask": coarse_mask, 
                      "final_mask": final_mask}

        # replace mask token with refined token
        queries = torch.cat([queries[:,0:1,:], output_refined_token, queries[:,2:,:]],dim=1)

        # obtain new point, including position and content
        if int(os.environ.get("CURRENT_EPOCH", 0))>4 or not self.training:
            point_sample_ref_map = unceratinty_map_norm * (torch.sigmoid(refined_mask)-torch.sigmoid(coarse_mask))
            point_sample_ref_map = point_sample_ref_map.flatten(1) # B × HW
            gumbel_dist = torch.distributions.gumbel.Gumbel(
                torch.tensor(0., device=point_sample_ref_map.device, dtype=point_sample_ref_map.dtype),
                torch.tensor(1., device=point_sample_ref_map.device, dtype=point_sample_ref_map.dtype))
            # gumbel softmax top-k
            if self.training:
                ret_p = self.gumbel_softmax_topk(gumbel_dist, point_sample_ref_map, n_sample_points) # B × HW
                ret_n = self.gumbel_softmax_topk(gumbel_dist, -point_sample_ref_map, n_sample_points)
            else:
                topk_values, topk_indices = torch.topk(point_sample_ref_map, k=n_sample_points, dim=1)
                khot = torch.zeros_like(point_sample_ref_map,memory_format=torch.legacy_contiguous_format)
                ret_p = khot.scatter_(1, topk_indices, 1)
                topk_values, topk_indices = torch.topk(-point_sample_ref_map, k=n_sample_points, dim=1)
                khot = torch.zeros_like(point_sample_ref_map,memory_format=torch.legacy_contiguous_format)
                ret_n = khot.scatter_(1, topk_indices, 1)
            ret_p = ret_p.view(b, 64, 64)
            ret_n = ret_n.view(b, 64, 64)
            device = point_sample_ref_map.device
            grid = torch.ones((64, 64), device=device, dtype=torch.float32)
            y_embed = grid.cumsum(dim=0) - 0.5
            x_embed = grid.cumsum(dim=1) - 0.5
            y_embed = y_embed / 64
            x_embed = x_embed / 64
            pe = prompt_encoder.pe_layer._pe_encoding(torch.stack([x_embed, y_embed], dim=-1)) # H x W x C  [position]
            content = F.interpolate(
                keys.permute(0,2,1).view(b, c, h, w),
                (64,64),
                mode="bilinear",
                align_corners=False,
            )
            content = (content + image_embedding.view(b,c,64,64)).permute(0,2,3,1) 
            sample_p_pe, sample_p_content = self.sample_points_promt(pe, content, ret_p)
            sample_n_pe, sample_n_content = self.sample_points_promt(pe, content, ret_n)
            sample_p_pe += prompt_encoder.point_embeddings[1].weight
            sample_n_pe += prompt_encoder.point_embeddings[0].weight
            # update point prompt
            query_pe = torch.cat([query_pe,sample_p_pe,sample_n_pe], dim=1)
            queries = torch.cat([queries,sample_p_content,sample_n_content], dim=1)

        # return queries, guiding_embedding, query_pe, mask, image_record.get("boxes", None), ret_p, ret_n
        return queries, guiding_embedding, query_pe, mask, None, None


    def gumbel_softmax_topk(self, gumbel_dist, input, n_sample_points=4, tau=1):
        EPSILON = np.finfo(np.float32).tiny
        gumbels = gumbel_dist.sample(input.shape)
        gumbels = (input + gumbels)
        khot = torch.zeros_like(gumbels,memory_format=torch.legacy_contiguous_format)
        onehot_approx = torch.zeros_like(gumbels)
        for i in range(n_sample_points):
            khot_mask = torch.max(1.0 - onehot_approx, torch.tensor([EPSILON]).cuda())
            gumbels = gumbels + torch.log(khot_mask)
            onehot_approx = F.softmax(gumbels / tau, dim=-1)
            khot = khot + onehot_approx
        khot_hard = torch.zeros_like(khot,memory_format=torch.legacy_contiguous_format)
        val, ind = torch.topk(khot, n_sample_points, dim=1)
        khot_hard = khot_hard.scatter_(1, ind, 1)
        res = khot_hard - khot.detach() + khot
        return res
    
    def sample_points_promt(self, pe, content, ret):
        pe = pe.unsqueeze(0) * ret.unsqueeze(-1)
        b, h, w, c = pe.shape
        sample_pe = pe[ret==1].reshape(b, -1, c)
        content = content * ret.unsqueeze(-1)
        sample_content = content[ret==1].reshape(b, -1, c)
        return  sample_pe, sample_content

class MaskDecoderPA(nn.Module):
    def __init__(self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        vit_dim: int = 1024,) ->None:
        super().__init__()

        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )
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

        #-----------------------image guiding---------------------------------
        self.guiding_conv = nn.Sequential(
            nn.Conv2d(4, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.Conv2d(transformer_dim // 4, transformer_dim, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim),
            nn.GELU(),
            nn.Conv2d(transformer_dim, transformer_dim, kernel_size=1),
        )
        #-----------------------prompt adapter-------------------------------
        self.prompt_adapter = PromptAdapater(transformer_dim,self.transformer.num_heads)



    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        interm_embeddings: torch.Tensor,
        hq_token_only: bool,
        prompt_encoder,
        input_images,
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
          torch.Tensor: batched predicted pa masks
        """

        vit_features = interm_embeddings[0].permute(0, 3, 1, 2) # early-layer ViT feature, after 1st global attention block in ViT
        hq_features=self.embedding_encoder(image_embeddings)+self.compress_vit_feat(vit_features)
        early_embeddings = vit_features
        final_embeddings = image_embeddings
        input_images = F.interpolate(
            input_images,
            (256,256),
            mode="bilinear",
            align_corners=False,
        )
        # import matplotlib.pyplot as plt
        # plt.imshow(input_images.detach().cpu().squeeze(0).permute(1, 2, 0))
        # plt.show()
            # generate guiding embedding
        input_images=input_images.squeeze(0)
        image_grad = misc.generalized_image_grad(input_images).unsqueeze(0)/255
        image_with_grad = torch.cat((input_images, image_grad), dim=0)
        guiding_embedding = self.guiding_conv(image_with_grad.unsqueeze(0))
        masks, iou_pred, interm_result = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            early_embeddings=early_embeddings,
            final_embeddings=final_embeddings,
            prompt_encoder=prompt_encoder,
            guiding_embedding=guiding_embedding,
            hq_feature = hq_features,
        )


        # Select the correct mask or masks for output
        if multimask_output:
            # mask with highest score
            mask_slice = slice(1,self.num_mask_tokens-1)
            iou_pred = iou_pred[:, mask_slice]
            iou_pred, max_iou_idx = torch.max(iou_pred,dim=1)
            iou_pred = iou_pred.unsqueeze(1)
            masks_multi = masks[:, mask_slice, :, :]
            masks_sam = masks_multi[torch.arange(masks_multi.size(0)),max_iou_idx].unsqueeze(1)
        else:
            # singale mask output, default
            mask_slice = slice(0, 1)
            iou_pred = iou_pred[:,mask_slice]
            masks_sam = masks[:,mask_slice]
        masks_hq = masks[:,slice(self.num_mask_tokens-1, self.num_mask_tokens), :, :]
        if hq_token_only:
            masks=masks_hq
        else:
             masks = masks_sam + masks_hq
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        early_embeddings: torch.Tensor,
        final_embeddings: torch.Tensor,
        hq_feature: torch.Tensor,
        prompt_encoder,
        guiding_embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""

        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight,self.hf_token.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0) 
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        prompt_adapter_args = {
            "early_embeddings": early_embeddings, "final_embeddings": final_embeddings, "prompt_encoder": prompt_encoder,  "prompt_adapter": self.prompt_adapter, "guiding_embedding":guiding_embedding}
        hs, src, Interm_result = self.transformer.forward_with_prompt_adapter(src, pos_src, tokens, prompt_adapter_args)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)

        upscaled_embedding_sam = self.output_upscaling(src)
        upscaled_embedding_ours = self.embedding_maskfeature(upscaled_embedding_sam) + hq_feature.repeat(b,1,1,1)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            if i < 4:
                hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
            else:
                hyper_in_list.append(self.hf_mlp(mask_tokens_out[:, i, :]))

        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding_sam.shape

        masks_sam = (hyper_in[:,:self.num_mask_tokens-1] @ upscaled_embedding_sam.view(b, c, h * w)).view(b, -1, h, w)
        masks_ours = (hyper_in[:,self.num_mask_tokens-1:] @ upscaled_embedding_ours.view(b, c, h * w)).view(b, -1, h, w)
        masks = torch.cat([masks_sam,masks_ours],dim=1)

        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred, Interm_result
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

