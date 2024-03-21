import cv2
import numpy
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import os
import random
from tqdm import tqdm
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
import argparse
from utils.utils import *
import time
from sklearn.model_selection import train_test_split
import albumentations as A
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
import os
import csv
from datetime import datetime
import xgboost as xgb
import seaborn as sns
import pandas as pd
from skimage import filters
from scipy.ndimage import label, find_objects

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class CustomDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        image = self.embeddings[idx]
        label = self.labels[idx]
        return image, label


def get_embedding(img, predictor):
    predictor.set_image(img)
    img_emb = predictor.get_image_embedding()
    return img_emb


def augment(image, mask):
    # Define an augmentation pipeline
    transform = A.Compose([
        A.Rotate(limit=30, p=0.5),  # Rotation
        A.RandomScale(scale_limit=0.2, p=0.5),  # Scaling
        A.GaussNoise(var_limit=(5, 25), p=0.5),  # Gaussian Noise
        A.GaussianBlur(blur_limit=(1, 5), p=0.5),  # Gaussian Blur
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),  # Brightness & Contrast
        A.RandomGamma(gamma_limit=(20, 60), p=0.5),  # Gamma Augmentation
        A.HorizontalFlip(p=0.5),  # Horizontal Mirroring
        A.VerticalFlip(p=0.5),  # Vertical Mirroring
    ])

    augmented = transform(image=image, mask=mask)
    return augmented['image'], augmented['mask']


def dice_coeff(pred, target):
    smooth = 1.
    # Ensure the inputs are numpy arrays. If they're PyTorch tensors, convert them to numpy arrays
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()

    # Flatten the arrays
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)

    intersection = (pred_flat * target_flat).sum()
    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def create_dataset_for_SVM(embeddings, labels):
    # Flatten the embeddings and labels to create a dataset for RF
    # embeddings shape: (N, 256, 64, 64) -> (N*64*64, 256)
    # labels shape: (N, 64, 64) -> (N*64*64,)
    N, C, H, W = embeddings.shape
    embeddings_flat = embeddings.reshape(-1, C)
    labels_flat = labels.reshape(-1)
    return embeddings_flat, labels_flat


def predict_and_reshape_otsu(model, X, val_labels, original_shape):
    otsu_median_thresh = []
    heatmaps = []

    # Loop over each example in the batch
    for i in range(original_shape[0]):
        image_flat = X[i].reshape(-1, X[i].shape[0])  # Ensure this reshapes correctly
        pred_probs_flat = model.predict_proba(image_flat)[:, 1]
        pred_probs = pred_probs_flat.reshape((64, 64))

        heatmap_normalized = cv2.normalize(pred_probs, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_normalized = np.uint8(heatmap_normalized)
        heatmaps.append(heatmap_normalized)
        median_filtered = cv2.medianBlur(heatmap_normalized, 3)
        _, otsu_thresh = cv2.threshold(median_filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours((otsu_thresh/255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize variables to store the best contour based on Dice score
        best_contour = None
        best_dice_score = -1

        # Calculate the Dice score for each contour
        for contour in contours:
            mask_temp = np.zeros_like(otsu_thresh)
            cv2.drawContours(mask_temp, [contour], -1, color=255, thickness=cv2.FILLED)
            mask_temp = (mask_temp / 255).astype(np.uint8)

            dice_score = dice_coeff_individual(mask_temp, val_labels[i])

            if dice_score > best_dice_score:
                best_dice_score = dice_score
                best_contour = contour

        # Create a mask for the contour with the highest Dice score
        mask_otsu = np.zeros_like(otsu_thresh)
        if best_contour is not None:
            cv2.drawContours(mask_otsu, [best_contour], -1, color=255, thickness=cv2.FILLED)
        mask_otsu = (mask_otsu / 255).astype(np.uint8)

        otsu_median_thresh.append(mask_otsu)

    return otsu_median_thresh, heatmaps


def predict_and_reshape(model, X, val_labels, original_shape):
    median_thresh = []


    # Loop over each example in the batch
    for i in range(original_shape[0]):
        image_flat = X[(i*4096):((i+1)*4096)]
        pred_probs_flat = model.predict(image_flat)
        pred_probs = pred_probs_flat.reshape((64, 64))

        heatmap_normalized = cv2.normalize(pred_probs, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_normalized = np.uint8(heatmap_normalized)
        median_filtered = cv2.medianBlur(heatmap_normalized, 3)
        _, otsu_thresh = cv2.threshold(median_filtered, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours((otsu_thresh/255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize variables to store the best contour based on Dice score
        best_contour = None
        best_dice_score = -1

        # Calculate the Dice score for each contour
        for contour in contours:
            mask_temp = np.zeros_like(otsu_thresh)
            cv2.drawContours(mask_temp, [contour], -1, color=255, thickness=cv2.FILLED)
            mask_temp = (mask_temp / 255).astype(np.uint8)

            dice_score = dice_coeff_individual(mask_temp, val_labels[i])

            if dice_score > best_dice_score:
                best_dice_score = dice_score
                best_contour = contour

        # Create a mask for the contour with the highest Dice score
        mask_otsu = np.zeros_like(otsu_thresh)
        if best_contour is not None:
            cv2.drawContours(mask_otsu, [best_contour], -1, color=255, thickness=cv2.FILLED)
        mask_otsu = (mask_otsu / 255).astype(np.uint8)

        median_thresh.append(mask_otsu)
    return median_thresh


def dice_coeff_individual(pred, target):
    smooth = 1.
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    intersection = (pred_flat * target_flat).sum()
    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def calculate_average_dice(pred_masks, true_masks):
    dice_scores = []
    for pred_mask, true_mask in zip(pred_masks, true_masks):
        dice_score = dice_coeff_individual(pred_mask, true_mask)
        dice_scores.append(dice_score)
    return np.mean(dice_scores)


def calculate_iou(ground_truth, prediction):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
    ground_truth (tuple): A tuple of (x1, y1, x2, y2) representing the bottom left and top right corners of the ground truth bounding box.
    prediction (tuple): A tuple of (x1, y1, x2, y2) representing the bottom left and top right corners of the predicted bounding box.

    Returns:
    float: The IoU score.
    """

    # Unpack the coordinates
    gt_x1, gt_y1, gt_x2, gt_y2 = ground_truth
    pred_x1, pred_y1, pred_x2, pred_y2 = prediction

    # Calculate the (x, y) coordinates of the intersection rectangle
    inter_x1 = max(gt_x1, pred_x1)
    inter_y1 = max(gt_y1, pred_y1)
    inter_x2 = min(gt_x2, pred_x2)
    inter_y2 = min(gt_y2, pred_y2)

    # Compute the area of intersection rectangle
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Compute the area of both the prediction and ground truth rectangles
    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)

    # Compute the union area by taking both areas and subtracting the intersection area
    union_area = gt_area + pred_area - inter_area

    # Compute the IoU by dividing the intersection area by the union area
    iou = inter_area / union_area

    return iou


def save_aggregated_metrics_with_std(all_metrics, all_metrics_otsu, all_metrics_SAM, all_metrics_SAM_point,
                                     all_metrics_SAM_GT, all_metrics_SAM_GTp, all_metrics_SAM_multi):
    # Define the model names for easier reference
    df_list = [pd.DataFrame(metrics) for metrics in
               [all_metrics, all_metrics_otsu, all_metrics_SAM, all_metrics_SAM_point, all_metrics_SAM_GT,
                all_metrics_SAM_GTp, all_metrics_SAM_multi]]
    model_names = ['Threshold', 'Otsu', 'SAM', 'SAM Point', 'SAM GT', 'SAM GT Point', 'Multi Point SAM']

    aggregated_metrics = []
    for i, df in enumerate(df_list):
        # Calculate mean and std dev for each metric
        metrics_mean = df.mean(axis=0)
        metrics_std = df.std(axis=0)

        # Prepare a dictionary to hold the aggregated metrics with mean and std dev
        aggregated_metric = {f"{metric}_mean": metrics_mean[metric] for metric in df.columns}
        aggregated_metric.update({f"{metric}_std": metrics_std[metric] for metric in df.columns})
        aggregated_metric['model'] = model_names[i]

        aggregated_metrics.append(aggregated_metric)

    # Save to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'model_metrics_aggregated_{timestamp}.csv'
    fieldnames = ['model'] + [f"{metric}_{stat}" for metric in df.columns for stat in ['mean', 'std']]

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for metrics in aggregated_metrics:
            writer.writerow(metrics)


def SAM_predict(predictor, image=None, bounding_box=None, point_prompt=None, heatmap=None):
    # Check if an image is provided and set it
    if image is not None:
        predictor.set_image(image)

    # Initialize variables for point_coords and point_labels
    input_point = None
    input_label = None

    # Check if a point prompt is provided
    if point_prompt is not None:
        # Assuming point_prompt is a tuple or list in the form (x, y, label)
        input_point = np.array([[point_prompt[0, 0], point_prompt[0, 1]]])
        input_label = np.array([point_prompt[0, 2]])

    # Call predictor's predict method with the provided or default parameters
    if bounding_box is not None:
        masks_pred, _, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=bounding_box[None, :],
            multimask_output=False,
            confidence_map=heatmap
        )
    else:
        masks_pred, _, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=None,
            multimask_output=False,
            confidence_map=heatmap
        )

    # Find contours
    mask = masks_pred[0].astype('uint8')
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour based on area
    largest_contour = max(contours, key=cv2.contourArea)

    # Create a mask for the largest contour
    mask_SAM = np.zeros_like(mask)
    cv2.drawContours(mask_SAM, [largest_contour], -1, color=1, thickness=cv2.FILLED)

    kernel = np.ones((10, 10), np.uint8)
    closing = cv2.morphologyEx(mask_SAM, cv2.MORPH_CLOSE, kernel)


    return closing, logits


def flatten_and_concatenate_arrays(array_list):
    """
    Flatten each array in the list and concatenate them into a single array.

    Parameters:
    array_list (list of np.array): List of arrays to be flattened and concatenated.

    Returns:
    np.array: A single flattened array containing all the elements from the input list.
    """
    # Flatten each array and store it in a new list
    flattened_arrays = [arr.flatten() for arr in array_list]

    # Concatenate all flattened arrays into one
    concatenated_array = np.concatenate(flattened_arrays)

    return concatenated_array


def visualize_predictions(org_img, images, masks, model, num_samples=3, val=False, eval_num=0):
    if len(images) < num_samples:
        num_samples = len(images)

    for i in range(num_samples):
        image = images[i]
        mask = masks[i]

        # Flatten the image for prediction
        image_flat = image.reshape(-1, image.shape[0])
        # Get prediction probabilities for the positive class
        pred_probs_flat = model.predict_proba(image_flat)[:, 1]
        # Reshape the prediction probabilities back to the original mask shape
        pred_probs = pred_probs_flat.reshape(mask.shape)
        # Normalize data to 0 and 255
        heatmap_normalized = cv2.normalize(pred_probs, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_normalized = np.uint8(heatmap_normalized)

        # Apply Gaussian and Median Filtering
        gaussian_filtered = cv2.GaussianBlur(heatmap_normalized, (5, 5), 0)
        median_filtered = cv2.medianBlur(heatmap_normalized, 5)

        # Edge detection Sobel
        edges_heat = filters.sobel(heatmap_normalized)
        edges_heat = cv2.normalize(edges_heat, None, 0, 255, cv2.NORM_MINMAX)
        edges_gaussian = filters.sobel(gaussian_filtered)
        edges_gaussian = cv2.normalize(edges_gaussian, None, 0, 255, cv2.NORM_MINMAX)
        edges_median = filters.sobel(median_filtered)
        edges_median = cv2.normalize(edges_median, None, 0, 255, cv2.NORM_MINMAX)
        edges_img = filters.sobel(org_img[i])
        edges_img = edges_img[:, :, 0] + edges_img[:, :, 1] + edges_img[:, :, 2]
        edges_img = cv2.normalize(edges_img, None, 0, 255, cv2.NORM_MINMAX)
        edges_img = filters.sobel(edges_img)
        edges_img = cv2.normalize(edges_img, None, 0, 255, cv2.NORM_MINMAX)

        # Combine heatmap with edge
        combo_heat = heatmap_normalized + (2 * edges_heat)
        combo_heat = cv2.normalize(combo_heat, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        combo_gaussian = gaussian_filtered + (2 * edges_gaussian)
        combo_gaussian = cv2.normalize(combo_gaussian, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        combo_median = median_filtered + (2 * edges_median)
        combo_median = cv2.normalize(combo_median, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

        # Apply threshold to the original and filtered heatmaps
        _, heatmap_thresh = cv2.threshold(heatmap_normalized, 127, 255, cv2.THRESH_BINARY)
        _, gaussian_thresh = cv2.threshold(gaussian_filtered, 127, 255, cv2.THRESH_BINARY)
        _, median_thresh = cv2.threshold(median_filtered, 127, 255, cv2.THRESH_BINARY)
        _, otsu_heatmap_thresh = cv2.threshold(heatmap_normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, otsu_gaussian_thresh = cv2.threshold(gaussian_filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, otsu_median_thresh = cv2.threshold(median_filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, combo_heat_thresh = cv2.threshold(combo_heat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, combo_gaussian_thresh = cv2.threshold(combo_gaussian, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, combo_median_thresh = cv2.threshold(combo_median, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours((otsu_median_thresh / 255).astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest contour based on area
        largest_contour = max(contours, key=cv2.contourArea)

        # Create a mask for the largest contour
        mask_otsu = np.zeros_like(otsu_median_thresh)
        cv2.drawContours(mask_otsu, [largest_contour], -1, color=255, thickness=cv2.FILLED)

        fig, axes = plt.subplots(8, 3, figsize=(15, 40))  # Adjusting figure size for better visibility

        # Original image and mask
        axes[0, 0].imshow(org_img[i])
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')

        axes[0, 1].imshow(mask, cmap='gray')
        axes[0, 1].set_title("True Mask")
        axes[0, 1].axis('off')

        axes[0, 2].imshow(edges_img, cmap='jet')
        axes[0, 2].set_title("Image Edges")
        axes[0, 2].axis('off')

        # Original Heatmap, Threshold, and Histogram
        axes[1, 0].imshow(pred_probs, cmap='jet')
        axes[1, 0].set_title("Original Heatmap")
        axes[1, 0].axis('off')

        axes[1, 1].imshow(heatmap_thresh, cmap='gray')
        axes[1, 1].set_title("Heatmap Threshold")
        axes[1, 1].axis('off')

        axes[1, 2].hist(pred_probs_flat, bins=50, color='blue', alpha=0.7, log=True)
        axes[1, 2].set_title("Heatmap Histogram")
        axes[1, 2].set_xlabel("Intensity")
        axes[1, 2].set_ylabel("Pixel Count")

        # Gaussian Filtered Heatmap, Threshold, and Histogram
        axes[2, 0].imshow(gaussian_filtered, cmap='jet')
        axes[2, 0].set_title("Gaussian Heatmap")
        axes[2, 0].axis('off')

        axes[2, 1].imshow(gaussian_thresh, cmap='gray')
        axes[2, 1].set_title("Gaussian Threshold")
        axes[2, 1].axis('off')

        axes[2, 2].hist(gaussian_filtered.ravel(), bins=50, color='green', alpha=0.7, log=True)
        axes[2, 2].set_title("Gaussian Histogram")
        axes[2, 2].set_xlabel("Intensity")
        axes[2, 2].set_ylabel("Pixel Count")

        # Median Filtered Heatmap, Threshold, and Histogram
        axes[3, 0].imshow(median_filtered, cmap='jet')
        axes[3, 0].set_title("Median Heatmap")
        axes[3, 0].axis('off')

        axes[3, 1].imshow(median_thresh, cmap='gray')
        axes[3, 1].set_title("Median Threshold")
        axes[3, 1].axis('off')

        axes[3, 2].hist(median_filtered.ravel(), bins=50, color='red', alpha=0.7, log=True)
        axes[3, 2].set_title("Median Histogram")
        axes[3, 2].set_xlabel("Intensity")
        axes[3, 2].set_ylabel("Pixel Count")

        axes[4, 0].imshow(otsu_heatmap_thresh, cmap='gray')
        axes[4, 0].set_title("Otsu Threshold")
        axes[4, 0].axis('off')

        axes[4, 1].imshow(otsu_gaussian_thresh, cmap='gray')
        axes[4, 1].set_title("Otsu Gaussian Threshold")
        axes[4, 1].axis('off')

        axes[4, 2].imshow(mask_otsu, cmap='gray')
        axes[4, 2].set_title("Otsu Median Threshold")
        axes[4, 2].axis('off')

        axes[5, 0].imshow(edges_heat, cmap='jet')
        axes[5, 0].set_title("Combo Heatmap")
        axes[5, 0].axis('off')

        axes[5, 1].imshow(edges_gaussian, cmap='jet')
        axes[5, 1].set_title("Combo Gaussian ")
        axes[5, 1].axis('off')

        axes[5, 2].imshow(edges_median, cmap='jet')
        axes[5, 2].set_title("Combo Median")
        axes[5, 2].axis('off')

        axes[6, 0].imshow(combo_heat, cmap='jet')
        axes[6, 0].set_title("Combo Heatmap")
        axes[6, 0].axis('off')

        axes[6, 1].imshow(combo_gaussian, cmap='jet')
        axes[6, 1].set_title("Combo Gaussian ")
        axes[6, 1].axis('off')

        axes[6, 2].imshow(combo_median, cmap='jet')
        axes[6, 2].set_title("Combo Median")
        axes[6, 2].axis('off')

        axes[7, 0].imshow(combo_heat_thresh, cmap='gray')
        axes[7, 0].set_title("Combo Threshold")
        axes[7, 0].axis('off')

        axes[7, 1].imshow(combo_gaussian_thresh, cmap='gray')
        axes[7, 1].set_title("Combo Gaussian Threshold")
        axes[7, 1].axis('off')

        axes[7, 2].imshow(combo_median_thresh, cmap='gray')
        axes[7, 2].set_title("Combo Median Threshold")
        axes[7, 2].axis('off')

        plt.tight_layout()
        if not val:
            plt.savefig(f"/content/visualisation/Fold{eval_num}-train_{i}.png")
        else:
            plt.savefig(f"/content/visualisation/Fold{eval_num}-val_{i}.png")
            # np.save(f"/content/image_{i}.npy", org_img)
            # np.save(f"/content/heatmap_{i}.npy", pred_probs)


def visualise_SAM(org_img, maskGT, thresh_mask, otsu_mask, SAM_mask, SAM_mask_GT, SAMp_mask, SAMp_mask_GT, otsu_BB,
                  thresh_BB, GT_BB, heatmap, points, pointsGT, logits_test):
    otsu_BB_resized = np.array(otsu_BB) / 16
    thresh_BB_resized = np.array(thresh_BB) / 16
    GT_BB_resized = np.array(GT_BB) / 16
    for i in range(len(org_img)):
        logit = cv2.normalize(logits_test[i], None, 0, 255, cv2.NORM_MINMAX)
        logit = 255 - logit
        logit = cv2.normalize(logit, None, 0, 255, cv2.NORM_MINMAX)

        # Create a mask where SAM_mask is 1
        SAM_mask_resized = cv2.resize(SAM_mask[i], dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
        masked_logit = np.where(SAM_mask_resized == 1, logit, 0)
        _, logit_thresh = cv2.threshold(masked_logit, 250, 255, cv2.THRESH_BINARY)

        # Apply Sobel edge filter to the masked logit image
        edges_logit = filters.sobel(masked_logit)
        edges_logit = cv2.normalize(edges_logit, None, 0, 255, cv2.NORM_MINMAX)

        heatmap_resized = cv2.resize(heatmap[i], dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
        combo = logit + 2 * heatmap_resized
        combo = cv2.normalize(combo, None, 0, 255, cv2.NORM_MINMAX)

        fig, axes = plt.subplots(2, 5, figsize=(25, 10))
        # Original image and mask
        axes[0, 0].imshow(org_img[i])
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')

        axes[0, 1].imshow(maskGT[i], cmap='gray')
        axes[0, 1].set_title("Ground Truth")
        axes[0, 1].axis('off')

        axes[0, 2].imshow(maskGT[i], cmap='gray')
        axes[0, 2].set_title("Ground Truth - Bounding Boxes")
        show_box(thresh_BB_resized[i], axes[0, 2], color='green')
        show_box(otsu_BB_resized[i], axes[0, 2], color='red')
        show_box(GT_BB_resized[i], axes[0, 2], color='blue')
        axes[0, 2].plot(points[i][0] / 16, points[i][1] / 16, 'r.')
        axes[0, 2].plot(pointsGT[i][0] / 16, pointsGT[i][1] / 16, 'g.')
        axes[0, 2].legend(['Threshold BB', 'Otsu BB', 'Ground Truth BB'])
        axes[0, 2].axis('off')

        axes[0, 3].imshow(heatmap[i], cmap='jet')
        axes[0, 3].set_title("Prediction Heatmap")
        axes[0, 3].axis('off')

        axes[0, 4].imshow(logit, cmap='jet')
        axes[0, 4].set_title("SAM Logits")
        axes[0, 4].axis('off')

        axes[1, 0].imshow(otsu_mask[i], cmap='gray')
        axes[1, 0].set_title("Otsu Threshold Mask")
        show_box(otsu_BB_resized[i], axes[1, 0])
        axes[1, 0].plot(points[i][0] / 16, points[i][1] / 16, 'r.')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(SAM_mask[i], cmap='gray')
        axes[1, 1].set_title("SAM Mask")
        show_box(otsu_BB[i], axes[1, 1])
        axes[1, 1].plot(points[i][0], points[i][1], 'r.')
        axes[1, 1].axis('off')
        plt.tight_layout()

        axes[1, 2].imshow(SAMp_mask[i], cmap='gray')
        axes[1, 2].set_title("SAM Mask - BB + Point")
        axes[1, 2].plot(points[i][0], points[i][1], 'g.')
        show_box(otsu_BB[i], axes[1, 2])
        axes[1, 2].axis('off')
        plt.tight_layout()

        axes[1, 3].imshow(SAM_mask_GT[i], cmap='gray')
        axes[1, 3].set_title("SAM Mask - Ground Truth BB")
        axes[1, 3].plot(pointsGT[i][0], pointsGT[i][1], 'g.')
        show_box(GT_BB[i], axes[1, 3])
        axes[1, 3].axis('off')
        plt.tight_layout()

        axes[1, 4].imshow(thresh_mask[i], cmap='gray')
        axes[1, 4].set_title("SAM Mask - BB + Point")
        # axes[1, 4].plot(pointsGT[i][0], pointsGT[i][1], 'g.')
        # show_box(GT_BB[i], axes[1, 4])
        axes[1, 4].axis('off')
        plt.tight_layout()

        # masked_logit_values = masked_logit.flatten()
        # masked_logit_values = masked_logit_values[masked_logit_values > 0]
        # axes[1, 4].hist(masked_logit_values, bins=500, color='blue', edgecolor='black')
        # axes[1, 4].set_title("Histogram of Masked Logit Values")
        # axes[1, 4].set_xlim([150, 255])  # Assuming logit values are normalized to 0-255
        # axes[1, 4].set_xlabel("Pixel Value")
        # axes[1, 4].set_ylabel("Frequency")
        # plt.tight_layout()

        plt.savefig(f"/content/visualisation/SAM segmentation {i}.png")


def monte_carlo_sample_from_mask(heatmap, mask, ground_truth_mask, num_fg = 10, num_bg = 10):
    """
    Samples points from the mask using Monte Carlo method and verifies them against the ground truth.

    Args:
    - heatmap (np.array): The heatmap from which to sample points.
    - mask (np.array): The binary mask to sample foreground and background points from.
    - ground_truth_mask (np.array): The ground truth binary mask to verify the sampled points.
    - base_n_points (int): Base number of points to sample.

    Returns:
    - tuple of (foreground_points, background_points, correct_positives, correct_negatives): Where each element is a list of tuples (x, y) for the points, and correct_positives and correct_negatives are counts of correctly sampled points.
    """
    foreground_indices = np.where(mask == 1)
    background_indices = np.where(mask == 0)

    # Adjusting n_points based on the size of the foreground and background
    foreground_pixel_count = len(foreground_indices[0])
    background_pixel_count = len(background_indices[0])

    n_points_foreground = min(num_fg, int(0.75 * foreground_pixel_count)) if foreground_pixel_count > 0 else 0
    n_points_background = min(num_bg, int(0.75 * background_pixel_count)) if background_pixel_count > 0 else 0

    # Sampling points
    foreground_probs = heatmap[foreground_indices]
    foreground_points = sample_points(foreground_indices, foreground_probs, n_points_foreground, alpha = 5)
    background_probs = heatmap[background_indices]
    background_points = sample_points(background_indices, background_probs, n_points_background, fg=False)

    # Check each point against the ground truth mask
    correct_positives = sum(ground_truth_mask[y, x] for x, y in foreground_points)
    correct_negatives = sum(1 - ground_truth_mask[y, x] for x, y in background_points)

    return foreground_points, background_points, correct_positives, correct_negatives, n_points_foreground, n_points_background


def sample_points(indices, probabilities, n_points, fg=True, alpha=10.0):
    """
    Helper function to perform sampling given indices and probabilities.
    :param indices: Tuple of numpy arrays, where each array corresponds to one of the dimensions.
                    For a 2D image, it should be (row_indices, col_indices).
    :param probabilities: Array of probabilities corresponding to each index.
    :param n_points: Number of points to sample.
    :param fg: Flag to indicate if sampling is for foreground (True) or background (False).
    :param alpha: Exponent to skew the probabilities. Higher than 1 makes high probs even higher and low probs lower.
    :return: List of sampled points.
    """
    if fg:
        if n_points == 0:
            return []
        # Apply transformation to skew the probabilities
        probabilities_transformed = np.power(probabilities, alpha)
        total_prob = probabilities_transformed.sum()
        if total_prob > 0:
            probabilities_normalized = probabilities_transformed / total_prob
            print('yay')
        else:
            probabilities_normalized = probabilities/probabilities.sum()
            print('boo')
        samples = np.random.choice(len(probabilities), size=n_points, replace=False, p=probabilities_normalized)
        return list(zip(indices[1][samples], indices[0][samples]))
    else:
        if n_points == 0:
            return []
        samples = np.random.choice(len(probabilities), size=n_points, replace=False)
        return list(zip(indices[1][samples], indices[0][samples]))



def sample_top_n_points(indices, probabilities, n_points):
    """
    Sample the n points with the highest probabilities.

    Args:
        indices (tuple of arrays): Tuple of numpy arrays containing the indices of the probabilities array.
        probabilities (np.array): Array containing the probabilities for each point.
        n_points (int): Number of top probabilities to select.

    Returns:
        list of tuples: List containing the coordinates of the top n points.
    """
    # Flatten the indices to make them align with the flattened probabilities array
    flattened_indices = np.vstack(indices).T  # This should result in a (num_points, 2) array

    # Sort probabilities and select top n
    top_n_indices = np.argsort(probabilities)[-n_points:]  # Get indices of top n probabilities

    # Select the corresponding top n points
    top_n_points = flattened_indices[top_n_indices]

    # Convert points back to list of tuples (x, y)
    top_n_points_list = [tuple(point) for point in top_n_points]

    return top_n_points_list


def dynamic_threshold_nms(indices, probabilities, n_points, min_distance, decay_factor=0.9):
    """
    Selects points dynamically based on a combination of high probability and spatial spreading.

    Args:
        indices (tuple of arrays): Indices of the grid or image.
        probabilities (np.array): Probabilities corresponding to each point.
        n_points (int): Number of points to select.
        min_distance (int): Minimum spatial distance between selected points.
        decay_factor (float): Factor by which to decay the threshold on each iteration. Closer to 1 is slower decay.

    Returns:
        list of tuples: Coordinates of the selected points.
    """
    # Flatten the indices to align with the probabilities array
    flattened_indices = np.vstack(indices).T
    sorted_idx = np.argsort(probabilities)[::-1]  # Indices sorted by decreasing probability

    selected_points = []
    current_threshold = probabilities[sorted_idx[0]]  # Start with the highest probability

    for idx in sorted_idx:
        if len(selected_points) >= n_points:
            break  # Stop if we have selected enough points

        point = flattened_indices[idx]
        prob = probabilities[idx]

        if prob < current_threshold:
            # Dynamically reduce the threshold
            current_threshold *= decay_factor

        # Check if the point is too close to any already selected point
        too_close = any(np.linalg.norm(point - p) < min_distance for p in selected_points)

        if not too_close and prob >= current_threshold:
            selected_points.append(point)

    return [tuple(point) for point in selected_points]


import cv2
import numpy as np


def visualize_and_save_points(image, mask, points, i, heatmap, SAM_pred, SAM_pred_multi, otsu_mask):
    """
    Visualizes points on the image and ground truth mask. Positive points are green,
    negative points are red. Points on the ground truth are circles; others are crosses.

    Args:
    - image (np.array): The original image.
    - mask (np.array): The ground truth mask.
    - points (list of tuples): Each tuple contains (x, y, positive, on_ground_truth),
      where 'positive' and 'on_ground_truth' are booleans.
    - output_path (str): Path to save the output image.
    """
    fig, ax = plt.subplots(2, 3, figsize=(18, 12))
    ax[0,0].imshow(image)
    ax[0,1].imshow(mask, cmap='gray')
    ax[0,2].imshow(heatmap, cmap='jet')
    ax[1,0].imshow(SAM_pred, cmap='gray')
    ax[1,1].imshow(SAM_pred_multi, cmap='gray')
    ax[1,2].imshow(otsu_mask, cmap='gray')

    for x, y, positive, on_ground_truth in points:
        color = 'green' if positive else 'red'
        marker = 'o' if on_ground_truth else 'x'
        ax[0,0].scatter(x, y, c=color, marker=marker)
        ax[0,1].scatter(x, y, c=color, marker=marker)
        ax[0,2].scatter(x, y, c=color, marker=marker)
        ax[1,2].scatter(x, y, c=color, marker=marker)

    ax[0,0].set_title('Original Image with Points')
    ax[0,1].set_title('Ground Truth Mask with Points')
    ax[0,2].set_title('Heatmap with Points')
    ax[1,0].set_title('SAM BB Prediction')
    ax[1,1].set_title('SAM Multi Prediction')
    ax[1,2].set_title('SAM Multi Prediction')
    ax[0,0].axis('off')
    ax[0,1].axis('off')
    ax[0,2].axis('off')
    ax[1,0].axis('off')
    ax[1,1].axis('off')
    ax[1,2].axis('off')

    plt.tight_layout()
    plt.savefig(f"/content/visualisation/multi point {i}.png")
    plt.close()


def train(args, predictor):
    all_metrics = []
    all_metrics_otsu = []
    all_metrics_SAM = []
    all_metrics_SAM_point = []
    all_metrics_SAM_GT = []
    all_metrics_SAM_GTp = []
    all_metrics_SAM_multi = []
    fg_accuracies = []
    bg_accuracies = []

    data_path = args.data_path
    assert os.path.exists(data_path), 'data path does not exist!'

    num_image = args.k

    fnames = os.listdir(os.path.join(data_path, 'images'))
    # get k random indices from fnames
    random.shuffle(fnames)
    val_fnames = fnames[-args.val_size:]
    fnames[-args.val_size:] = []

    # create a number of different training sets
    train_fnames = []
    for i in range(args.evaluation_num):
        segment = fnames[(i * num_image):(i + 1) * num_image]
        train_fnames.append(segment)

    # image augmentation and embedding processing
    num_augmentations = int(args.augmentation_num)  # Number of augmented versions to create per image

    def process_images(file_names, augment_data=True):
        image_embeddings = []
        labels = []
        org_img = []
        original_sizes = []
        maskOrgs = []

        def process_and_store(img, msk):
            original_size = [msk.shape[1], msk.shape[0]]
            original_sizes.append(original_size)
            # Resize and process the mask and image
            maskOrgs.append(msk)
            resized_mask = cv2.resize(msk, dsize=(64, 64), interpolation=cv2.INTER_NEAREST)
            # Find contours
            contours, _ = cv2.findContours(resized_mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

            # Create a mask for the largest contour
            mask_large = np.zeros_like(resized_mask)
            for contour in contours:
                if cv2.contourArea(contour) > 5:
                    cv2.drawContours(mask_large, [contour], -1, color=255, thickness=cv2.FILLED)

            resized_mask = (mask_large / 255).astype(np.uint8)
            resized_img = cv2.resize(img, dsize=(1024, 1024), interpolation=cv2.INTER_NEAREST)

            # Process the image to create an embedding
            img_emb = get_embedding(resized_img, predictor)
            img_emb = img_emb.cpu().numpy().transpose((2, 0, 3, 1)).reshape((256, 64, 64))
            image_embeddings.append(img_emb)
            labels.append(resized_mask)
            org_img.append(resized_img)

        for fname in tqdm(file_names):
            # Read data
            image = cv2.imread(os.path.join(data_path, 'images', fname))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(os.path.join(data_path, 'masks', fname), cv2.IMREAD_GRAYSCALE)
            _, mask = cv2.threshold(mask, 128, 1, cv2.THRESH_BINARY)

            if augment_data:
                process_and_store(image, mask)
                for _ in range(num_augmentations):
                    # Apply augmentations
                    augmented_image, augmented_mask = augment(image, mask)
                    process_and_store(augmented_image, augmented_mask)
            else:
                # For validation data, do not apply augmentation
                process_and_store(image, mask)

        return image_embeddings, labels, org_img, original_sizes, maskOrgs

    # Process validation images without augmentation
    val_embeddings, val_labels, val_images, val_sizes, val_masks = process_images(val_fnames, augment_data=False)

    # Convert to tensor
    val_embeddings_tensor = torch.stack([torch.Tensor(e) for e in val_embeddings])
    val_labels_tensor = torch.stack([torch.Tensor(l) for l in val_labels])

    val_embeddings_flat, val_labels_flat = create_dataset_for_SVM(val_embeddings_tensor.numpy(),
                                                                  val_labels_tensor.numpy())
    for i in range(args.evaluation_num):
        # Process training images with augmentation
        train_embeddings, train_labels, train_images, train_sizes, train_masks = process_images(train_fnames[i],
                                                                                                augment_data=True)

        # Convert to tensors
        train_embeddings_tensor = torch.stack([torch.Tensor(e) for e in train_embeddings])
        train_labels_tensor = torch.stack([torch.Tensor(l) for l in train_labels])

        # Use the same function as defined for Random Forest
        train_embeddings_flat, train_labels_flat = create_dataset_for_SVM(train_embeddings_tensor.numpy(),
                                                                          train_labels_tensor.numpy())

        # Perform oversampling on the training data
        ros = RandomOverSampler(random_state=42)
        train_embeddings_oversampled, train_labels_oversampled = ros.fit_resample(train_embeddings_flat,
                                                                                  train_labels_flat)

        # Initialize the XGBoost classifier model
        model = xgb.XGBClassifier(objective='binary:logistic', colsample_bytree=0.54, gamma=0.46, learning_rate=0.0383,
                                  subsample=0.8, booster='gbtree',
                                  max_depth=8, min_child_weight=4, alpha=10, n_estimators=550, reg_alpha=0.474,
                                  reg_lambda=0.098, verbosity=2, device="cuda")
        model.fit(train_embeddings_flat, train_labels_flat)

        # Predict on the validation set
        start_time = time.time()  # Start timing
        predicted_masks_svm = predict_and_reshape(model, val_embeddings_flat, val_labels, (len(val_embeddings_tensor), 64, 64))
        end_time = time.time()  # End timing
        prediction_time = (end_time - start_time) / 25
        pred_original = predicted_masks_svm
        pred_original_resized_eval = []
        for mask in range(len(pred_original)):
            resized_mask = cv2.resize(pred_original[mask], dsize=val_sizes[mask],
                                      interpolation=cv2.INTER_NEAREST)
            pred_original_resized_eval.append(resized_mask)

        # Predict on the validation set (OTSU)
        start_time = time.time()  # Start timing
        predicted_masks_otsu, heatmaps = predict_and_reshape_otsu(model, val_embeddings, val_labels,
                                                                  (len(val_embeddings_tensor), 64, 64))
        end_time = time.time()  # End timing
        prediction_time_otsu = (end_time - start_time) / 25
        otsu_original = predicted_masks_otsu
        otsu_original_resized_eval = []
        for mask in range(len(otsu_original)):
            resized_mask = cv2.resize(otsu_original[mask], dsize=val_sizes[mask],
                                      interpolation=cv2.INTER_NEAREST)
            otsu_original_resized_eval.append(resized_mask)

        # Define the kernel for dilation
        kernel = np.ones((2, 2), np.uint8)

        # predicted_masks_svm = cv2.dilate(predicted_masks_svm, kernel, iterations=3)
        # predicted_masks_svm = cv2.erode(predicted_masks_svm, kernel, iterations=3)

        # Evaluate the SVM model
        # accuracy_svm = accuracy_score(val_labels_flat, predicted_masks_svm.reshape(-1))
        # print(f'SVM Accuracy (Dilation + Erosion): {accuracy_svm}')
        # print(classification_report(val_labels_flat, predicted_masks_svm.reshape(-1)))

        # prompt the sam with the bounding box
        BBIoUs = []
        BBIoUOtsu = []
        BBoxes = []
        BBoxes_Otsu = []
        BBoxes_GT = []
        points_GT = []
        points_otsu = []

        # resize the masks for bounding boxes
        predicted_masks_svm_resized = [None] * len(predicted_masks_svm)
        otsu_original_resized = [None] * len(otsu_original)
        val_labels_resized = [None] * len(val_labels)

        for j in range(len(predicted_masks_svm)):
            predicted_masks_svm_resized[j] = cv2.resize(predicted_masks_svm[j], dsize=(1024, 1024),
                                                        interpolation=cv2.INTER_NEAREST)
            otsu_original_resized[j] = cv2.resize(otsu_original[j], dsize=(1024, 1024),
                                                  interpolation=cv2.INTER_NEAREST)
            val_labels_resized[j] = cv2.resize(val_labels[j], dsize=(1024, 1024),
                                               interpolation=cv2.INTER_NEAREST)

        for j in range(len(predicted_masks_svm)):
            H, W = predicted_masks_svm_resized[j].shape
            y_indices, x_indices = np.where(predicted_masks_svm_resized[j] > 0)
            y_otsu, x_otsu = np.where(otsu_original_resized[j] > 0)
            y_val, x_val = np.where(val_labels_resized[j] > 0)

            point_GT = get_max_dist_point(val_labels_resized[j])
            point_otsu = get_max_dist_point(otsu_original_resized[j])
            points_GT.append(point_GT)
            points_otsu.append(point_otsu)

            # Initialize default bounding box values
            bbox_default = np.array([0, 0, W, H])

            # Check if arrays are empty and handle accordingly
            if x_val.size > 0 and y_val.size > 0:
                x_minVal, x_maxVal = np.min(x_val), np.max(x_val)
                y_minVal, y_maxVal = np.min(y_val), np.max(y_val)
                bboxVal = np.array([x_minVal, y_minVal, x_maxVal, y_maxVal])
            else:
                bboxVal = bbox_default

            if x_indices.size > 0 and y_indices.size > 0:
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                bbox = np.array([x_min, y_min, x_max, y_max])
            else:
                bbox = bbox_default

            if x_otsu.size > 0 and y_otsu.size > 0:
                x_minOtsu, x_maxOtsu = np.min(x_otsu), np.max(x_otsu)
                y_minOtsu, y_maxOtsu = np.min(y_otsu), np.max(y_otsu)
                bboxOtsu = np.array([x_minOtsu, y_minOtsu, x_maxOtsu, y_maxOtsu])
            else:
                bboxOtsu = bbox_default

            BBIoU = calculate_iou(bboxVal, bbox)
            BBoxes.append(bbox)
            BBoxes_GT.append(bboxVal)
            BBIoUs.append(BBIoU)
            BBoxes_Otsu.append(bboxOtsu)
            BBIoU = calculate_iou(bboxVal, bboxOtsu)
            BBIoUOtsu.append(BBIoU)

        for k in range(len(heatmaps)):
            heatmaps[k] = cv2.medianBlur(heatmaps[k], 3)

        # Get evaluations from SAM
        print('Evaluating using SAM', i)
        SAM_pred = []
        SAM_pred_resized = []
        prediction_time_SAM = 0
        for j in range(len(val_images)):
            input_point = np.array([[points_otsu[j][0], points_otsu[j][1], 1]])
            start_time = time.time()  # Start timing
            masks_pred, logits = SAM_predict(predictor, val_images[j], bounding_box=BBoxes_Otsu[j], point_prompt=None, heatmap=None)
            mask_SAM = masks_pred.astype('uint8')
            mask_SAM_resized = cv2.resize(mask_SAM, dsize=val_sizes[j], interpolation=cv2.INTER_NEAREST)
            end_time = time.time()  # End timing
            prediction_time_SAM += (end_time - start_time)
            SAM_pred.append(mask_SAM)
            SAM_pred_resized.append(mask_SAM_resized)
        prediction_time_SAM /= len(val_images)
        print('Finished SAM')

        print('Evaluating using SAM - Point', i)
        SAM_point_pred = []
        SAM_point_pred_resized = []
        prediction_time_SAM_point = 0
        logits_test = []

        for j in range(len(val_images)):
            input_point = np.array([[points_otsu[j][0], points_otsu[j][1], 1]])
            start_time = time.time()  # Start timing
            # masks_pred, logits = SAM_predict(predictor, val_images[j], bounding_box=BBoxes_Otsu[j],
            #                                  point_prompt=input_point,  heatmap=heatmaps[j])
            masks_pred, logits = SAM_predict(predictor, val_images[j], bounding_box=BBoxes_Otsu[j],
                                             point_prompt=None,  heatmap=None)
            mask_SAM = masks_pred.astype('uint8')
            logit_test = logits[0].astype('uint8')
            mask_SAM_resized = cv2.resize(mask_SAM, dsize=val_sizes[j], interpolation=cv2.INTER_NEAREST)
            end_time = time.time()  # End timing
            prediction_time_SAM_point += (end_time - start_time)
            SAM_point_pred.append(mask_SAM)
            SAM_point_pred_resized.append(mask_SAM_resized)
            logits_test.append(logit_test)
        prediction_time_SAM_point /= len(val_images)
        print('Finished SAM')

        if i == 0:
            # Get evaluations from SAM
            print('Evaluating using SAM Ground Truth', i)
            SAM_pred_GT = []
            SAM_pred_GT_resized = []
            prediction_time_SAM_GT = 0
            for j in range(len(val_images)):
                start_time = time.time()  # Start timing
                masks_pred, logits = SAM_predict(predictor, val_images[j], bounding_box=BBoxes_GT[j],
                                                 point_prompt=None, heatmap=heatmaps[j])
                mask_SAM = masks_pred.astype('uint8')
                mask_SAM_GT_resized = cv2.resize(mask_SAM, dsize=val_sizes[j], interpolation=cv2.INTER_NEAREST)
                end_time = time.time()  # End timing
                prediction_time_SAM_GT += (end_time - start_time)
                SAM_pred_GT.append(mask_SAM)
                SAM_pred_GT_resized.append(mask_SAM_GT_resized)
            prediction_time_SAM_GT /= len(val_images)
            report_SAM_GT = classification_report(flatten_and_concatenate_arrays(val_masks),
                                                  flatten_and_concatenate_arrays(SAM_pred_GT_resized),
                                                  target_names=['0', '1'], output_dict=True)
            print('Finished SAM')

        if i == 0:
            # Get evaluations from SAM
            print('Evaluating using SAM Ground Truth - Point', i)
            SAM_pred_GTp = []
            SAM_pred_GTp_resized = []
            prediction_time_SAM_GTp = 0
            for j in range(len(val_images)):
                input_point = np.array([[points_GT[j][0], points_GT[j][1], 1]])
                start_time = time.time()  # Start timing
                masks_pred, logits = SAM_predict(predictor, val_images[j], bounding_box=BBoxes_GT[j],
                                                 point_prompt=input_point, heatmap=heatmaps[j])
                mask_SAM = masks_pred.astype('uint8')
                mask_SAM_GTp_resized = cv2.resize(mask_SAM, dsize=val_sizes[j], interpolation=cv2.INTER_NEAREST)
                end_time = time.time()  # End timing
                prediction_time_SAM_GTp += (end_time - start_time)
                SAM_pred_GTp.append(mask_SAM)
                SAM_pred_GTp_resized.append(mask_SAM_GTp_resized)
            prediction_time_SAM_GTp /= len(val_images)
            report_SAM_GTp = classification_report(flatten_and_concatenate_arrays(val_masks),
                                                   flatten_and_concatenate_arrays(SAM_pred_GTp_resized),
                                                   target_names=['0', '1'], output_dict=True)
            print('Finished SAM')

        print('Multi Point SAM')
        SAM_pred_multi = []
        SAM_pred_multi_resized = []
        prediction_time_SAM_multi = 0

        for j in range(len(val_images)):
            # Assuming heatmap and mask are available for each image
            coarse_mask = otsu_original_resized[j]  # Your method to obtain the coarse mask
            gt_mask = val_labels_resized[j]
            heatmap_resized = cv2.resize(heatmaps[j], dsize=(1024, 1024),
                                                        interpolation=cv2.INTER_NEAREST)
            foreground_points, background_points, correct_fg, correct_bg, num_points_fg, num_points_bg = monte_carlo_sample_from_mask(heatmap_resized, coarse_mask, gt_mask, args.num_fg, args.num_bg)

            fg_accuracy = correct_fg / num_points_fg
            fg_accuracies.append(fg_accuracy)
            bg_accuracy = correct_bg / num_points_bg
            bg_accuracies.append(bg_accuracy)

            # Combining foreground and background points
            combined_points = np.array(foreground_points + background_points)
            point_labels = np.array(
                [1] * len(foreground_points) + [0] * len(background_points))  # 1 for foreground, 0 for background

            if len(combined_points) > 0:
                input_points = np.hstack([combined_points, point_labels[:, None]])  # Reshape for SAM_predict
            else:
                input_points = None  # Handle the case where no points are available

            combined_points = [(x, y, True, val_labels_resized[j][y, x] == 1) for x, y in foreground_points] + \
                              [(x, y, False, val_labels_resized[j][y, x] == 0) for x, y in background_points]

            start_time = time.time()
            if input_points is not None:
                masks_pred, logits = SAM_predict(predictor, val_images[j], bounding_box=BBoxes_Otsu[j],
                                                 point_prompt=input_points, heatmap=heatmaps[j])
            else:
                masks_pred, logits = SAM_predict(predictor, val_images[j], bounding_box=BBoxes_Otsu[j], heatmap=heatmaps[j])
            end_time = time.time()
            prediction_time_SAM_multi += (end_time - start_time)

            mask_SAM = masks_pred.astype('uint8')
            if i == 0:
                visualize_and_save_points(val_images[j], val_labels_resized[j], combined_points,
                                      j, heatmap_resized, SAM_pred[j], mask_SAM, otsu_original_resized[j])
            mask_SAM_resized = cv2.resize(mask_SAM, dsize=val_sizes[j], interpolation=cv2.INTER_NEAREST)
            SAM_pred_multi.append(mask_SAM)
            SAM_pred_multi_resized.append(mask_SAM_resized)
        print('Finished SAM')

        # Evaluate the SVM model
        report = classification_report(flatten_and_concatenate_arrays(val_masks),
                                       flatten_and_concatenate_arrays(pred_original_resized_eval),
                                       target_names=['0', '1'], output_dict=True)
        report_otsu = classification_report(flatten_and_concatenate_arrays(val_masks),
                                            flatten_and_concatenate_arrays(otsu_original_resized_eval),
                                            target_names=['0', '1'], output_dict=True)
        report_SAM = classification_report(flatten_and_concatenate_arrays(val_masks),
                                           flatten_and_concatenate_arrays(SAM_pred_resized),
                                           target_names=['0', '1'], output_dict=True)
        report_SAM_point = classification_report(flatten_and_concatenate_arrays(val_masks),
                                                 flatten_and_concatenate_arrays(SAM_point_pred_resized),
                                                 target_names=['0', '1'], output_dict=True)
        report_SAM_multi = classification_report(flatten_and_concatenate_arrays(val_masks),
                                                 flatten_and_concatenate_arrays(SAM_pred_multi_resized),
                                                 target_names=['0', '1'], output_dict=True)


        # Dice Scores

        svm_dice_val = calculate_average_dice(pred_original_resized_eval, val_masks)
        otsu_dice_val = calculate_average_dice(otsu_original_resized_eval, val_masks)
        SAM_dice_val = calculate_average_dice(SAM_pred_resized, val_masks)
        SAM_point_dice_val = calculate_average_dice(SAM_point_pred_resized, val_masks)
        SAMGT_dice_val = calculate_average_dice(SAM_pred_GT_resized, val_masks)
        SAMGTp_dice_val = calculate_average_dice(SAM_pred_GTp_resized, val_masks)
        SAM_multi_dice_val = calculate_average_dice(SAM_pred_multi_resized, val_masks)

        metrics = {
            'eval_num': i,  # Evaluation number or model identifier
            'accuracy': report['accuracy'],
            'negative_precision': report['0']['precision'],
            'positive_precision': report['1']['precision'],
            'negative_recall': report['0']['recall'],
            'positive_recall': report['1']['recall'],
            'f1_score': report['weighted avg']['f1-score'],
            'BB IoU': np.mean(BBIoUs),
            'Time per Sample': prediction_time,
            'dice_score': svm_dice_val
        }
        all_metrics.append(metrics)

        metrics_otsu = {
            'eval_num': i,  # Evaluation number or model identifier
            'accuracy': report_otsu['accuracy'],
            'negative_precision': report_otsu['0']['precision'],
            'positive_precision': report_otsu['1']['precision'],
            'negative_recall': report_otsu['0']['recall'],
            'positive_recall': report_otsu['1']['recall'],
            'f1_score': report_otsu['weighted avg']['f1-score'],
            'BB IoU': np.mean(BBIoUOtsu),
            'Time per Sample': prediction_time_otsu,
            'dice_score': otsu_dice_val
        }
        all_metrics_otsu.append(metrics_otsu)

        metrics_SAM = {
            'eval_num': i,  # Evaluation number or model identifier
            'accuracy': report_SAM['accuracy'],
            'negative_precision': report_SAM['0']['precision'],
            'positive_precision': report_SAM['1']['precision'],
            'negative_recall': report_SAM['0']['recall'],
            'positive_recall': report_SAM['1']['recall'],
            'f1_score': report_SAM['weighted avg']['f1-score'],
            'BB IoU': np.mean(BBIoUOtsu),
            'Time per Sample': prediction_time_SAM,
            'dice_score': SAM_dice_val
        }
        all_metrics_SAM.append(metrics_SAM)

        metrics_SAM_point = {
            'eval_num': i,  # Evaluation number or model identifier
            'accuracy': report_SAM_point['accuracy'],
            'negative_precision': report_SAM_point['0']['precision'],
            'positive_precision': report_SAM_point['1']['precision'],
            'negative_recall': report_SAM_point['0']['recall'],
            'positive_recall': report_SAM_point['1']['recall'],
            'f1_score': report_SAM_point['weighted avg']['f1-score'],
            'BB IoU': np.mean(BBIoUOtsu),
            'Time per Sample': prediction_time_SAM_point,
            'dice_score': SAM_point_dice_val
        }
        all_metrics_SAM_point.append(metrics_SAM_point)

        metrics_SAM_GT = {
            'eval_num': i,  # Evaluation number or model identifier
            'accuracy': report_SAM_GT['accuracy'],
            'negative_precision': report_SAM_GT['0']['precision'],
            'positive_precision': report_SAM_GT['1']['precision'],
            'negative_recall': report_SAM_GT['0']['recall'],
            'positive_recall': report_SAM_GT['1']['recall'],
            'f1_score': report_SAM_GT['weighted avg']['f1-score'],
            'BB IoU': np.mean(BBIoUOtsu),
            'Time per Sample': prediction_time_SAM_GT,
            'dice_score': SAMGT_dice_val
        }
        all_metrics_SAM_GT.append(metrics_SAM_GT)

        metrics_SAM_GTp = {
            'eval_num': i,  # Evaluation number or model identifier
            'accuracy': report_SAM_GTp['accuracy'],
            'negative_precision': report_SAM_GTp['0']['precision'],
            'positive_precision': report_SAM_GTp['1']['precision'],
            'negative_recall': report_SAM_GTp['0']['recall'],
            'positive_recall': report_SAM_GTp['1']['recall'],
            'f1_score': report_SAM_GTp['weighted avg']['f1-score'],
            'BB IoU': np.mean(BBIoUOtsu),
            'Time per Sample': prediction_time_SAM_GTp,
            'dice_score': SAMGTp_dice_val
        }
        all_metrics_SAM_GTp.append(metrics_SAM_GTp)

        metrics_SAM_multi = {
            'eval_num': i,  # Evaluation number or model identifier
            'accuracy': report_SAM_multi['accuracy'],
            'negative_precision': report_SAM_multi['0']['precision'],
            'positive_precision': report_SAM_multi['1']['precision'],
            'negative_recall': report_SAM_multi['0']['recall'],
            'positive_recall': report_SAM_multi['1']['recall'],
            'f1_score': report_SAM_multi['weighted avg']['f1-score'],
            'BB IoU': np.mean(BBIoUOtsu),
            'Time per Sample': prediction_time_SAM_multi,
            'dice_score': SAM_multi_dice_val
        }
        all_metrics_SAM_multi.append(metrics_SAM_multi)

        # Visualize SVM predictions on the validation dataset
        # print("Validation Predictions with SVM:")
        if i == 0:
            # visualize_predictions(train_images, train_embeddings, train_labels, model, num_samples=25, val=False, eval_num=i)
            visualize_predictions(val_images, val_embeddings, val_labels, model, num_samples=25, val=True, eval_num=i)
            visualise_SAM(val_images, val_labels, pred_original, otsu_original, SAM_pred, SAM_pred_GT, SAM_point_pred,
                          SAM_pred_GTp, BBoxes_Otsu, BBoxes, BBoxes_GT, heatmaps, points_otsu, points_GT, logits_test)

    save_aggregated_metrics_with_std(all_metrics, all_metrics_otsu, all_metrics_SAM, all_metrics_SAM_point,
                                     all_metrics_SAM_GT, all_metrics_SAM_GTp, all_metrics_SAM_multi)

    print('Foreground Point Accuracy:', np.mean(fg_accuracies))
    print('Background Point Accuracy:', np.mean(bg_accuracies))
    print('Test')

    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--k', type=int, default=10, help='number of pics')
    parser.add_argument('--data_path', type=str, default='./data/Kvasir-SEG', help='path to train data')
    parser.add_argument('--model_type', type=str, default='vit_b', help='SAM model type')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/sam_vit_b_01ec64.pth', help='SAM checkpoint')
    parser.add_argument('--visualize', type=bool, default=True, help='visualize the results')
    parser.add_argument('--save_path', type=str, default='./results', help='path to save the results')
    parser.add_argument('--visualize_num', type=int, default=30, help='number of pics to visualize')
    parser.add_argument('--epochs', type=int, default=50, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate for the optimizer')
    parser.add_argument('--threshold', type=float, default=0.5, help='threshold for binary segmentation')
    parser.add_argument('--augmentation_num', type=float, default=20, help='number of image augmentations to perform')
    parser.add_argument('--evaluation_num', type=int, default=5, help='number of models to trian for evaluation')
    parser.add_argument('--val_size', type=int, default=25, help='number of validation images')
    parser.add_argument('--num_fg', type=int, default=25, help='number of selected foreground points')
    parser.add_argument('--num_bg', type=int, default=25, help='number of selected background points')
    args = parser.parse_args()

    # set random seed
    random.seed(42)

    # register the SAM model
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(args.device)
    global predictor
    predictor = SamPredictor(sam)
    print('SAM model loaded!', '\n')

    if args.visualize:
        model = train(args, predictor)
    else:
        test(args, predictor)


if __name__ == '__main__':
    main()