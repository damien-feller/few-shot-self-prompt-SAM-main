import cv2
import numpy
import torch
import torch.nn.functional as F
import torch.nn as nn
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
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
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

def predict_and_reshape(model, X, original_shape):
    predictions = model.predict(X)
    return predictions.reshape(original_shape)


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

        # Apply threshold to the filtered heatmaps
        _, gaussian_thresh = cv2.threshold(gaussian_filtered, 127, 255, cv2.THRESH_BINARY)
        _, median_thresh = cv2.threshold(median_filtered, 127, 255, cv2.THRESH_BINARY)

        fig, axes = plt.subplots(3, 4, figsize=(20, 15))  # Adjusting figure size for better visibility

        # Original image and mask
        axes[0, 0].imshow(org_img[i])
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')

        axes[0, 1].imshow(mask, cmap='gray')
        axes[0, 1].set_title("True Mask")
        axes[0, 1].axis('off')

        # Prediction Heat Map
        im = axes[0, 2].imshow(pred_probs, cmap='jet')
        fig.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04, label='Probability')
        axes[0, 2].set_title("Prediction Heat Map")
        axes[0, 2].axis('off')

        # Histogram of Prediction Probabilities
        axes[0, 3].hist(pred_probs_flat, bins=50, color='blue', alpha=0.7, log=True)
        axes[0, 3].set_title("Probability Histogram")
        axes[0, 3].set_xlabel("Probability")
        axes[0, 3].set_ylabel("Pixel Count")

        # Gaussian Filtered Heatmap and Threshold
        axes[1, 0].imshow(gaussian_filtered, cmap='gray')
        axes[1, 0].set_title("Gaussian Filtered")
        axes[1, 0].axis('off')

        axes[1, 1].imshow(gaussian_thresh, cmap='gray')
        axes[1, 1].set_title("Gaussian Threshold")
        axes[1, 1].axis('off')

        # Median Filtered Heatmap and Threshold
        axes[1, 2].imshow(median_filtered, cmap='gray')
        axes[1, 2].set_title("Median Filtered")
        axes[1, 2].axis('off')

        axes[1, 3].imshow(median_thresh, cmap='gray')
        axes[1, 3].set_title("Median Threshold")
        axes[1, 3].axis('off')

        # Original Adaptive and Fixed Thresholds
        adaptive_thresh = cv2.adaptiveThreshold(heatmap_normalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY, 11, 2)
        fixed_thresh = (pred_probs > 0.5).astype(np.uint8)

        axes[2, 0].imshow(fixed_thresh, cmap='gray')
        axes[2, 0].set_title("Fixed Threshold = 0.5")
        axes[2, 0].axis('off')

        axes[2, 1].imshow(adaptive_thresh, cmap='gray')
        axes[2, 1].set_title("Adaptive Threshold")
        axes[2, 1].axis('off')

        plt.tight_layout()
        if not val:
            plt.savefig(f"/content/visualisation/Fold{eval_num}-train_{i}.png")
        else:
            plt.savefig(f"/content/visualisation/Fold{eval_num}-val_{i}.png")



def train(args, predictor):
    all_metrics = []
    feature_importance = []
    data_path = args.data_path
    assert os.path.exists(data_path), 'data path does not exist!'

    num_image = args.k

    fnames = os.listdir(os.path.join(data_path, 'images'))
    # get k random indices from fnames
    random.shuffle(fnames)
    val_fnames = fnames[-25:]
    fnames[-25:] = []


    #create a number of different training sets
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

        def process_and_store(img, msk):
            # Resize and process the mask and image
            resized_mask = cv2.resize(msk, dsize=(64, 64), interpolation=cv2.INTER_NEAREST)
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

        return image_embeddings, labels, org_img

    # Process validation images without augmentation
    val_embeddings, val_labels, val_images = process_images(val_fnames, augment_data=False)

    # Convert to tensor
    val_embeddings_tensor = torch.stack([torch.Tensor(e) for e in val_embeddings])
    val_labels_tensor = torch.stack([torch.Tensor(l) for l in val_labels])

    val_embeddings_flat, val_labels_flat = create_dataset_for_SVM(val_embeddings_tensor.numpy(),
                                                                 val_labels_tensor.numpy())
    for i in range(args.evaluation_num):
        # Process training images with augmentation
        train_embeddings, train_labels, train_images = process_images(train_fnames[i], augment_data=True)

        # Convert to tensors
        train_embeddings_tensor = torch.stack([torch.Tensor(e) for e in train_embeddings])
        train_labels_tensor = torch.stack([torch.Tensor(l) for l in train_labels])

        # Use the same function as defined for Random Forest
        train_embeddings_flat, train_labels_flat = create_dataset_for_SVM(train_embeddings_tensor.numpy(),
                                                                         train_labels_tensor.numpy())

        # Perform oversampling on the training data
        ros = RandomOverSampler(random_state=42)
        train_embeddings_oversampled, train_labels_oversampled = ros.fit_resample(train_embeddings_flat, train_labels_flat)

        # Initialize the XGBoost classifier model
        model = xgb.XGBClassifier(objective='binary:logistic', colsample_bytree=0.3, learning_rate=0.1,
                                  max_depth=5, alpha=10, n_estimators=100, verbosity=2, device = "cuda")
        model.fit(train_embeddings_flat, train_labels_flat)

        # Predict on the validation set
        start_time = time.time()  # Start timing
        predicted_masks_svm = predict_and_reshape(model, val_embeddings_flat, (len(val_embeddings_tensor), 64, 64))
        predicted_masks_svm = (predicted_masks_svm > args.threshold).astype(np.uint8)
        end_time = time.time()  # End timing
        prediction_time = (end_time - start_time) / 25
        pred_original =predicted_masks_svm

        # Define the kernel for dilation
        kernel = np.ones((2, 2), np.uint8)

        # predicted_masks_svm = cv2.dilate(predicted_masks_svm, kernel, iterations=3)
        # predicted_masks_svm = cv2.erode(predicted_masks_svm, kernel, iterations=3)

        # Evaluate the SVM model
        #accuracy_svm = accuracy_score(val_labels_flat, predicted_masks_svm.reshape(-1))
        # print(f'SVM Accuracy (Dilation + Erosion): {accuracy_svm}')
        # print(classification_report(val_labels_flat, predicted_masks_svm.reshape(-1)))

        # prompt the sam with the bounding box
        BBIoUs = []
        for j in range(len(predicted_masks_svm)):
            H, W = predicted_masks_svm[j].shape
            y_indices, x_indices = np.where(predicted_masks_svm[j] > 0)
            y_val, x_val = np.where(val_labels[j] > 0)
            if np.all(predicted_masks_svm[j] == 0):
                bbox = np.array([0, 0, H, W])
            else:
                x_minVal, x_maxVal = np.min(x_val), np.max(x_val)
                y_minVal, y_maxVal = np.min(y_val), np.max(y_val)
                # x_minVal = max(0, x_minVal - np.random.randint(0, 20))
                # x_max = min(W, x_max + np.random.randint(0, 20))
                # y_min = max(0, y_min - np.random.randint(0, 20))
                # y_max = min(H, y_max + np.random.randint(0, 20))
                bboxVal = np.array([x_minVal, y_minVal, x_maxVal, y_maxVal])

                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                # x_min = max(0, x_min - np.random.randint(0, 20))
                # x_max = min(W, x_max + np.random.randint(0, 20))
                # y_min = max(0, y_min - np.random.randint(0, 20))
                # y_max = min(H, y_max + np.random.randint(0, 20))
                bbox = np.array([x_min, y_min, x_max, y_max])
                BBIoU = calculate_iou(bboxVal, bbox)
                BBIoUs.append(BBIoU)


        # Evaluate the SVM model
        report = classification_report(val_labels_flat, pred_original.reshape(-1),target_names = ['0','1'], output_dict=True)
        #accuracy_svm = accuracy_score(val_labels_flat, pred_original.reshape(-1))
        # print(f'SVM Accuracy: {accuracy_svm}')
        # predicted_masks_train = predict_and_reshape(model, train_embeddings_flat, (len(train_embeddings_tensor), 64, 64))
        # predicted_masks_train = (predicted_masks_train > args.threshold).astype(np.uint8)
        # print(classification_report(train_labels_flat, predicted_masks_train.reshape(-1)))

        # Dice Scores
        # svm_dice_val = dice_coeff(torch.Tensor(predicted_masks_svm), torch.Tensor(val_labels))
        # print('SVM Dice (Dilation + Erosion): ', svm_dice_val)
        svm_dice_val = dice_coeff(torch.Tensor(np.array(pred_original)), torch.Tensor(np.array(val_labels)))
        #print('SVM Dice: ', svm_dice_val)

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
            'dice_score': svm_dice_val.numpy()
        }
        all_metrics.append(metrics)

        # Visualize SVM predictions on the validation dataset
        #print("Validation Predictions with SVM:")
        if i == 0:
            visualize_predictions(train_images, train_embeddings, train_labels, model, num_samples=25, val=False, eval_num=i)
            visualize_predictions(val_images, val_embeddings, val_labels, model, num_samples=25, val=True, eval_num=i)


    # Define the file path, e.g., by including a timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'/content/model_metrics_{timestamp}.csv'

    # Check if the file exists to write headers only once
    file_exists = os.path.isfile(filename)

    with open(filename, 'w', newline='') as csvfile:  # Note: using 'w' to overwrite or create new
        fieldnames = ['eval_num', 'accuracy', 'negative_precision', 'positive_precision',
                      'negative_recall', 'positive_recall', 'f1_score','BB IoU','Time per Sample', 'dice_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()  # Write the header

        for metrics in all_metrics:
            writer.writerow(metrics)  # Write each model's metrics

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