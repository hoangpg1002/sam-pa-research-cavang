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
from segment_anything_training.modeling import TwoWayTransformer, MaskDecoder

from utils.dataloader import get_im_gt_name_dict, create_dataloaders, RandomHFlip, Resize, LargeScaleJitter
from utils.losses import loss_masks, loss_masks_whole, loss_masks_whole_uncertain, loss_boxes, loss_uncertain, loss_iou
from utils.function import show_heatmap, show_anns, show_heatmap_ax, show_anns_ax, show_mask, show_points, show_box, show_only_points, compute_iou, compute_boundary_iou
import utils.misc as misc

from model.mask_decoder_pa_o import MaskDecoderPA

import logging
import csv
import time

import warnings
warnings.filterwarnings('ignore')


def main(net, train_datasets, valid_datasets, ):
    ### --- Step 1: Train or Valid dataset ---

    print("--- create training dataloader ---")
    train_im_gt_list = get_im_gt_name_dict(train_datasets, flag="train")
    train_dataloaders, train_datasets = create_dataloaders(train_im_gt_list,
                                                    my_transforms = [
                                                                RandomHFlip(),
                                                                LargeScaleJitter()
                                                                ],
                                                    batch_size = 4,
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
    

 
    ### --- Step 3: Train or Evaluate ---
    print("--- define optimizer ---")
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10)
    lr_scheduler.last_epoch = 0
    #train(net,optimizer, train_dataloaders, valid_dataloaders, lr_scheduler)
    sam = sam_model_registry["vit_l"](checkpoint="/kaggle/working/training/pretrained_checkpoint/sam_vit_l_01ec64.pth").to(device="cuda")
    evaluate(net, sam, valid_dataloaders)

def train(net, optimizer, train_dataloaders, valid_dataloaders, lr_scheduler):
    if misc.is_main_process():
        os.makedirs("train", exist_ok=True)
    epoch_start = 0
    epoch_num = 20
    train_num = len(train_dataloaders)

    net.train()
    _ = net.to(device="cuda")
    for n,p in net.named_parameters():
        if p.requires_grad:
            print(n)
    sam = sam_model_registry["vit_l"](checkpoint="/kaggle/working/training/pretrained_checkpoint/sam_vit_b_01ec64.pth")
    _ = sam.to(device="cuda")
    #sam = torch.nn.parallel.DistributedDataParallel(sam, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
    
    for epoch in range(epoch_start,epoch_num): 
        print("epoch:   ",epoch, "  learning rate:  ", optimizer.param_groups[0]["lr"])
        os.environ["CURRENT_EPOCH"] = str(epoch)
        metric_logger = misc.MetricLogger(delimiter="  ")

        for data in metric_logger.log_every(train_dataloaders, 50):
            inputs, labels = data['image'], data['label']
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            imgs = inputs.permute(0, 2, 3, 1).cpu().numpy()
            
            # input prompt
            input_keys = ['box','point','noise_mask','box+point','box+noise_mask','point+noise_mask','box+point+noise_mask']
            labels_box = misc.masks_to_boxes(labels[:,0,:,:])
            try:
                labels_points = misc.masks_sample_points(labels[:,0,:,:])
            except:
                # less than 10 points
                input_keys = ['box','noise_mask','box+noise_mask']
            labels_256 = F.interpolate(labels, size=(256, 256), mode='bilinear')
            labels_noisemask = misc.masks_noise(labels_256)

            batched_input = []
            gt_boxes = []
            for b_i in range(len(imgs)):
                dict_input = dict()
                input_image = torch.as_tensor(imgs[b_i].astype(dtype=np.uint8), device=sam.device).permute(2, 0, 1).contiguous()
                dict_input['image'] = input_image 
                input_type = random.choice(input_keys)
                gt_boxes.append((labels_box[b_i:b_i+1]/1024).clamp(min=0.0, max=1.0))  
                noise_box = misc.box_noise(labels_box[b_i:b_i+1], box_noise_scale=1)
                if  'box' in input_type:    
                    dict_input['boxes'] = labels_box[b_i:b_i+1] 
                elif 'point' in input_type:   
                    point_coords = labels_points[b_i:b_i+1]
                    dict_input['point_coords'] = point_coords
                    dict_input['point_labels'] = torch.ones(point_coords.shape[1], device=point_coords.device)[None,:]
                elif 'noise_mask' in input_type:   
                    dict_input['mask_inputs'] = labels_noisemask[b_i:b_i+1]
                else:
                    raise NotImplementedError
                dict_input['original_size'] = imgs[b_i].shape[:2]
                dict_input['label'] = labels[b_i:b_i+1]
                batched_input.append(dict_input)

            with torch.no_grad():
                batched_output, interm_embeddings = sam.forward_for_prompt_adapter(batched_input, multimask_output=False)
            
            gt_boxes = torch.cat(gt_boxes, 0)
            batch_len = len(batched_output)
            encoder_embedding = torch.cat([batched_output[i_l]['encoder_embedding'] for i_l in range(batch_len)], dim=0)
            image_pe = [batched_output[i_l]['image_pe'] for i_l in range(batch_len)]
            sparse_embeddings = [batched_output[i_l]['sparse_embeddings'] for i_l in range(batch_len)]
            dense_embeddings = [batched_output[i_l]['dense_embeddings'] for i_l in range(batch_len)]
            image_record = [batched_output[i_l]['image_record'] for i_l in range(batch_len)]
            input_images = batched_output[0]['input_images']

            masks_sam, iou_preds, uncertain_maps, final_masks, coarse_masks, refined_masks, box_preds = net(
                image_embeddings=encoder_embedding,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                interm_embeddings=interm_embeddings,
                image_record=image_record,
                prompt_encoder=sam.prompt_encoder,
                input_images=input_images
            )

            loss_mask, loss_dice = loss_masks_whole(masks_sam, labels/255.0, len(masks_sam)) 
            loss = loss_mask + loss_dice

            loss_mask_final, loss_dice_final = loss_masks_whole_uncertain(coarse_masks, refined_masks, labels/255.0, uncertain_maps, len(final_masks))
            loss = loss + (loss_mask_final + loss_dice_final)     
            loss_uncertain_map, gt_uncertain = loss_uncertain(uncertain_maps, labels)  
            loss = loss + loss_uncertain_map

            loss_dict = {"loss_mask": loss_mask, "loss_dice":loss_dice, 
                               "loss_mask_final": loss_mask_final, "loss_dice_final": loss_dice_final, 
                               "loss_uncertain_map": loss_uncertain_map}

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
        test_stats = evaluate(net, sam, valid_dataloaders)
        train_stats.update(test_stats)
        
        net.train()  

        if epoch % 1 == 0:
            model_name = "/epoch_"+str(epoch)+".pth"
            print('come here save at', "train" + model_name)
            misc.save_on_master(net.state_dict(), "train" + model_name)
    
    # Finish training
    print("Training Reaches The Maximum Epoch Number")

def evaluate(net, sam, valid_dataloaders):
    
    net.eval()
    net.to(device="cuda")
    print("Validating...")
    test_stats = {}

    for k in range(len(valid_dataloaders)):
        metric_logger = misc.MetricLogger(delimiter="  ")
        valid_dataloader = valid_dataloaders[k]
        print('valid_dataloader len:', len(valid_dataloader))
        
        iou_result = []
        biou_result = []
        img_id = []
        dataset_name = ['DIS','COIFT','HRSOD','ThinObject']
        total_time = 0

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
                dict_input['label'] = data_val['label'][b_i:b_i+1]
                batched_input.append(dict_input)

            with torch.no_grad():
                batched_output, interm_embeddings = sam.forward_for_prompt_adapter(batched_input, multimask_output=False)
            
            batch_len = len(batched_output)
            encoder_embedding = torch.cat([batched_output[i_l]['encoder_embedding'] for i_l in range(batch_len)], dim=0)
            image_pe = [batched_output[i_l]['image_pe'] for i_l in range(batch_len)]
            sparse_embeddings = [batched_output[i_l]['sparse_embeddings'] for i_l in range(batch_len)]
            dense_embeddings = [batched_output[i_l]['dense_embeddings'] for i_l in range(batch_len)]
            image_record = [batched_output[i_l]['image_record'] for i_l in range(batch_len)]
            input_images = batched_output[0]['input_images']

            masks_sam, iou_preds, uncertain_maps, final_masks, coarse_masks, refined_masks, box_preds = net(
                image_embeddings=encoder_embedding,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                interm_embeddings=interm_embeddings,
                image_record=image_record,
                prompt_encoder=sam.prompt_encoder,
                input_images=input_images
            )


            iou = compute_iou(masks_sam,labels_ori)
            boundary_iou = compute_boundary_iou(masks_sam,labels_ori)
            


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
    dataset_dis_local = {"name": "DIS5K-TR",
                 "im_dir": r"D:\StableDiffusion\sam-hq\data\DIS5K\DIS5K\DIS-VD\im",
                 "gt_dir": r"D:\StableDiffusion\sam-hq\data\DIS5K\DIS5K\DIS-VD\gt",
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

    #train_datasets = [dataset_dis_local]
    #valid_datasets = [dataset_dis_local] 
    train_datasets = [dataset_dis, dataset_thin, dataset_fss, dataset_duts, dataset_duts_te, dataset_ecssd, dataset_msra]
    valid_datasets = [dataset_dis_val, dataset_coift_val, dataset_hrsod_val, dataset_thin_val] 
    net = MaskDecoderPA("vit_l",is_train=True) 

    main(net, train_datasets, valid_datasets)
