
#! WIP -- don't push to public yet

import argparse
import csv
import math
import random
import traceback
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import wandb
from facenet_pytorch import InceptionResnetV1
from rtpt import RTPT
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt

from Plug_and_Play_Attacks.attacks.final_selection import perform_final_selection
from Plug_and_Play_Attacks.attacks.optimize import Optimization
from Plug_and_Play_Attacks.datasets.custom_subset import ClassSubset


from model_inversion.metrics.mi_classification_acc import ClassificationAccuracy
from model_inversion.metrics.mi_fid_score import FID_Score
from model_inversion.metrics.mi_prdc import PRDC
from model_inversion.metrics.mi_distance_metrics import DistanceEvaluation

from Plug_and_Play_Attacks.utils.datasets import (create_target_dataset, get_facescrub_idx_to_class,
                            get_stanford_dogs_idx_to_class)
# from Plug_and_Play_Attacks.utils.stylegan import create_image, load_discrimator, load_generator
from Plug_and_Play_Attacks.utils.wandb import *

from data_processing.datasets import get_datasets
from data_processing.data_augmentation import get_transforms
from model_inversion.plug_and_play.pnp_utils import *

from model_inversion.plug_and_play.fid_by_target import FID_Score_by_target

def evaluate_attack(evaluation_model,
                config,
                target_config,
                attack_img_dataset, # expects images to be cropped and resized properly before being passed
                targets, # corresponding to the images
                device,
                gpu_devices,
                idx_to_class,
                batch_size,
                batch_size_single,
                run_id,
                target_dataset,
                target_model=None,
                rtpt=None):
    

    ####################################
    #         Attack Accuracy          #
    ####################################

    evaluation_model = torch.nn.DataParallel(evaluation_model)
    evaluation_model.to(device)
    evaluation_model.eval()
    class_acc_evaluator = ClassificationAccuracy(evaluation_model,
                                                    device=device)

    # Compute attack accuracy on filtered samples
    acc_top1, acc_top5, predictions, avg_correct_conf, avg_total_conf, target_confidences, maximum_confidences, precision_list = class_acc_evaluator.compute_acc(
        attack_img_dataset,
        targets,
        # synthesis,
        # config,
        batch_size=batch_size * 2,
        # resize=299,
        rtpt=rtpt)
    
    # log precision list
    filename_precision = write_precision_list(
        f'model_inversion/plug_and_play/results/precision_list_filtered/{run_id}',
        precision_list)
    if config.training.wandb.track:
        wandb.save(filename_precision)

    print(
        f'Evaluation of {final_w.shape[0]} images on Inception-v3: \taccuracy@1={acc_top1:4f}, ',
        f'accuracy@5={acc_top5:4f}, correct_confidence={avg_correct_conf:4f}, total_confidence={avg_total_conf:4f}'
    )

    ####################################
    #    FID Score and GAN Metrics     #
    ####################################

    fid_score = None
    precision, recall = None, None
    density, coverage = None, None
    try:
        # set transformations
        # crop_size = config.attack_center_crop
        
        ### This is replaced by our dataload functions ###
        # target_transform = T.Compose([
        #     T.ToTensor(),
        #     T.Resize((299, 299), antialias=True),
        #     T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        # ])
        
        # training_dataset = create_target_dataset(target_dataset,
        #                                          target_transform)        
        
        target_transform_list = [T.Resize((299, 299), antialias=True)] #! TODO: Use parameters
        
        # This is to only use the transformation shown above
        target_config_no_aug = target_config
        target_config_no_aug.dataset.augment_data = False
        
        target_transform  = get_transforms(target_config_no_aug, target_transform_list, train=False) # Train is false to not get the data augmentations
        training_dataset, _, _ = get_datasets(target_config, train_transform= target_transform, test_transform=None)
        
        # create datasets
        # attack_dataset = TensorDataset(attack_imgs, targets)
        # attack_dataset.targets = targets #! might need this in some form
        
        training_dataset = ClassSubset(     
            training_dataset,
            target_classes=torch.unique(targets).cpu().tolist())

        # compute FID score #! TODO: Use parameters
        fid_evaluation = FID_Score(training_dataset, 
                                   attack_dataset,
                                   device=device,
                                #    crop_size=crop_size,
                                #    generator=synthesis,
                                   batch_size=batch_size * 3,
                                   dims=2048,
                                   num_workers=8,
                                   gpu_devices=gpu_devices)
        fid_score = fid_evaluation.compute_fid(rtpt)
        print(
            f'FID score computed on {final_w.shape[0]} attack samples and {config.dataset.name}: {fid_score:.4f}'
        )
                

        # compute precision, recall, density, coverage #! TODO: Use parameters
        prdc = PRDC(training_dataset,
                    attack_dataset,
                    device=device,
                    # crop_size=crop_size,
                    # generator=synthesis,
                    batch_size=batch_size * 3,
                    dims=2048,
                    num_workers=8,
                    gpu_devices=gpu_devices)
        precision, recall, density, coverage = prdc.compute_metric(
            num_classes=config.dataset.n_classes, k=3, rtpt=rtpt)
        print(
            f' Precision: {precision:.4f}, Recall: {recall:.4f}, Density: {density:.4f}, Coverage: {coverage:.4f}'
        )

        del training_dataset

    except Exception:
        print(traceback.format_exc())

    ####################################
    #         Feature Distance         #
    ####################################
    avg_dist_inception = None
    avg_dist_facenet = None
        
    if isinstance(evaluation_model, torch.nn.DataParallel):
        evaluation_model_dist = evaluation_model.module.model.feature_extractor
    else:
        evaluation_model_dist = evaluation_model.model.feature_extractor
    
    evaluation_model_dist.to(device)
    evaluation_model_dist.eval()

    ### Get our dataset
    target_transform_list = [T.Resize((299, 299), antialias=True)] #! TODO: Use parameters TODO: Maybe add cropping
    
    # This is to only use the transformation shown above
    target_config_no_aug = target_config
    target_config_no_aug.dataset.augment_data = False
    
    target_transform  = get_transforms(config=target_config_no_aug, 
                                        extra_augmentations=target_transform_list, 
                                        train=False)
    
    training_dataset_inception, _, _ = get_datasets(config=target_config, 
                                            train_transform=target_transform,
                                            test_transform=None)


    # Compute average feature distance on Inception-v3
    evaluate_inception = DistanceEvaluation(evaluation_model_dist, 299, training_dataset_inception, config.training.seed) # TODO: use parameter
    
    avg_dist_inception, mean_distances_list_inception = evaluate_inception.compute_dist(
        final_w,
        targets,
        batch_size=batch_size_single * 5,
        rtpt=rtpt)

    filename_distance = write_precision_list(
                        f'model_inversion/plug_and_play/results/distance_inceptionv3_list_filtered/{run_id}',
                        mean_distances_list_inception)
    
    if config.training.wandb.track:
        wandb.save(filename_distance)

    print('Mean Distance on Inception-v3: ',
            avg_dist_inception.cpu().item())

    del training_dataset_inception
    
    # Compute feature distance only for facial images # TODO: Still not modified
    if target_config.dataset.face_dataset:
        # Load FaceNet model for face recognition
        facenet = InceptionResnetV1(pretrained='vggface2')
        facenet = torch.nn.DataParallel(facenet, device_ids=gpu_devices)
        facenet.to(device)
        facenet.eval()
        
        ### Get our dataset
        face_net_img_size = 160 #TODO: use parameter
        
        target_transform_list = [T.Resize((face_net_img_size, face_net_img_size), antialias=True)]
        
        # This is to only use the transformation shown above
        target_config_no_aug = target_config
        target_config_no_aug.dataset.augment_data = False
        
        target_transform  = get_transforms(config=target_config_no_aug, 
                                            extra_augmentations=target_transform_list, 
                                            train=False)
        
        training_dataset_facenet, _, _ = get_datasets(config=target_config, 
                                                        train_transform=target_transform, 
                                                        test_transform=None)
            
        # Compute average feature distance on vggface
        evaluater_facenet = DistanceEvaluation(facenet, face_net_img_size, training_dataset_facenet, config.training.seed)
        
        
        avg_dist_facenet, mean_distances_list_facenet = evaluater_facenet.compute_dist(
            attack_imgs,
            targets,
            batch_size=batch_size_single * 8, #? Hardcoded?
            rtpt=rtpt)

        filename_distance = write_precision_list(
                            f'model_inversion/plug_and_play/results/distance_facenet_list_filtered/{run_id}',
                            mean_distances_list_facenet)
        wandb.save(filename_distance)
        print('Mean Distance on FaceNet: ', avg_dist_facenet.cpu().item())

        del training_dataset_facenet

        if target_model is not None:
            # Compute average feature distance on target model
            if isinstance(target_model, torch.nn.DataParallel):
                target_model_dist = target_model.module.feature_extractor
            else:
                target_model_dist = target_model.feature_extractor
            target_model_dist.to(device)
            target_model_dist.eval()
            
            ### Get our dataset
            target_img_size = 224 #TODO: use parameter
            
            target_transform_list = [T.Resize((target_img_size, target_img_size), antialias=True)]
            
            # This is to only use the transformation shown above
            target_config_no_aug = target_config
            target_config_no_aug.dataset.augment_data = False
            
            target_transform  = get_transforms(config=target_config_no_aug, 
                                                extra_augmentations=target_transform_list, 
                                                train=False)
            
            training_dataset_target_model, _, _ = get_datasets(config=target_config, 
                                                            train_transform=target_transform, 
                                                            test_transform=None)
                
            # Compute average feature distance on target model
            evaluater_target_model = DistanceEvaluation(target_model_dist, target_img_size, training_dataset_target_model, config.training.seed)
            avg_dist_target_model, mean_distances_list_target_model = evaluater_target_model.compute_dist(
                                                                        attack_imgs,
                                                                        targets,
                                                                        batch_size=batch_size_single * 8, #? Hardcoded?
                                                                        rtpt=rtpt)

            # make sure directory exists
            os.makedirs('model_inversion/plug_and_play/results/distance_target_model_list_filtered', exist_ok=True)
            filename_distance = write_precision_list(
                                f'model_inversion/plug_and_play/results/distance_target_model_list_filtered/{run_id}',
                                mean_distances_list_target_model)
            wandb.save(filename_distance)
            print('Mean Distance on target model: ', avg_dist_target_model.cpu().item())

            del training_dataset_target_model


    ####################################
    #          Finish Logging          #
    ####################################

    if rtpt:
        rtpt.step(subtitle=f'Finishing up')

    # Logging of final results
    print('Finishing attack, logging results and creating sample images.')
    num_classes = 10 #? Hard coded?
    num_imgs = 8     #? Hard coded?
    # Sample final images from the first and last classes
    label_subset = set(
        list(set(targets.tolist()))[:int(num_classes / 2)] +
        list(set(targets.tolist()))[-int(num_classes / 2):])
    log_imgs = []
    log_targets = []
    log_predictions = []
    log_max_confidences = []
    log_target_confidences = []
    # Log images with smallest feature distance
    for label in label_subset:
        mask = torch.where(targets == label, True, False)
        attack_imgs_masked = attack_imgs[mask][:num_imgs]
        # imgs = create_image(w_masked,
        #                     synthesis,
        #                     crop_size=config.attack_center_crop,
        #                     resize=config.attack_resize)
        log_imgs.append(attack_imgs_masked)
        log_targets += [label for i in range(num_imgs)]
        log_predictions.append(torch.tensor(predictions)[mask][:num_imgs])
        log_max_confidences.append(
            torch.tensor(maximum_confidences)[mask][:num_imgs])
        log_target_confidences.append(
            torch.tensor(target_confidences)[mask][:num_imgs])

    log_imgs = torch.cat(log_imgs, dim=0)
    log_predictions = torch.cat(log_predictions, dim=0)
    log_max_confidences = torch.cat(log_max_confidences, dim=0)
    log_target_confidences = torch.cat(log_target_confidences, dim=0)

    log_final_images(log_imgs, log_predictions, log_max_confidences,
                        log_target_confidences, idx_to_class)

    # Find closest training samples to final results
    log_nearest_neighbors(log_imgs,
                            log_targets,
                            evaluation_model_dist,
                            'InceptionV3',
                            target_dataset,
                            img_size=299,
                            seed=config.training.seed)

    # Use FaceNet only for facial images
    facenet = InceptionResnetV1(pretrained='vggface2')
    # facenet = torch.nn.DataParallel(facenet, device_ids=gpu_devices) #! Putting FaceNet on DataParallel currently causes "NCCL Error 3: internal error", not sure why
    facenet.to(device)
    facenet.eval()
    
    if target_config.dataset.face_dataset:
        log_nearest_neighbors(log_imgs,
                                log_targets,
                                facenet,
                                'FaceNet',
                                target_dataset,
                                img_size=160,
                                seed=config.training.seed)


    # Plot the m 
    distances_arr_inception = np.array([mean_distances_list_inception[i][1] for i in range(1,len(mean_distances_list_inception))])
    
    bins = 20
    plt = plot_histogram(distances_arr_inception, bins, 'Distribution of Feature Distances (Inception)', 
            'Feature Distance', 'Frequency', color='lightcoral')
    
    if config.training.wandb.track:
        # Log the plot directly to wandb
        wandb.log({"Distances Histogram (Inception)": wandb.Image(plt)})
    
    if target_config.dataset.face_dataset:
        plt.clf()
        distances_arr_facenet = np.array([mean_distances_list_facenet[i][1] for i in range(1,len(mean_distances_list_facenet))])
        
        plt = plot_histogram(distances_arr_facenet, bins, 'Distribution of Feature Distances (FaceNet)',
                                'Feature Distance', 'Frequency', color='skyblue')

        if config.training.wandb.track:
            # Log the plot directly to wandb
            wandb.log({"Distances Histogram (FaceNet)": wandb.Image(plt)})

    # Confidence histogram
    plt.clf()
    conf_bins = 30
    plt = plot_histogram(target_confidences, conf_bins, 'Distribution of Target Confidences',
                            'Confidence', 'Frequency', color='lightgreen')
    wandb.log({"Confidence Histogram": wandb.Image(plt)})

    # Final logging
    final_wandb_logging(avg_correct_conf, avg_total_conf, acc_top1,
                        acc_top5, avg_dist_facenet, avg_dist_inception,
                        fid_score, precision, recall, density, coverage)
        

def plot_histogram(data, bins, title, xlabel, ylabel, color):
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, edgecolor='black', color=color, alpha=0.8)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.8)
    plt.tight_layout()
    return plt
