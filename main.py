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

# class EfficientNetSegmentation(nn.Module):
#     def __init__(self, num_classes):
#         super(EfficientNetSegmentation, self).__init__()
#         # Custom layer to adapt the 256-channel input (LOSS OF INFORMATION???)
#         self.input_adaptation = nn.Conv2d(256, 3, 1)  # Convolution to convert from 256 to 3 channels
#
#         # Load pre-trained EfficientNet model
#         self.backbone = models.efficientnet_v2_l(weights='DEFAULT')
#
#         # Remove the average pooling and fully connected layer
#         self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
#
#         # Add a convolution layer to get the segmentation map
#         self.conv = nn.Conv2d(1280, num_classes, 1)
#
#         # Upsample to the desired output size
#         self.upsample = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=True)
#
#     def forward(self, x):
#         x = self.input_adaptation(x)
#         x = self.backbone(x) # Now x has the shape [batch_size, 2048, H, W]
#         x = self.conv(x)     # Convolution to get the segmentation map
#         x = self.upsample(x) # Upsample to the original image size
#         return x


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

def visualize_predictions(images, masks, model, num_samples=3, val=False):
    if len(images) < num_samples:
        num_samples = len(images)

    indices = np.random.choice(range(len(images)), num_samples, replace=False)

    for i in indices:
        image = images[i]
        mask = masks[i]

        # Flatten the image for SVM prediction
        image_flat = image.reshape(-1, image.shape[0])
        pred_flat = model.predict(image_flat)
        # Reshape the prediction to the original mask shape
        pred = pred_flat.reshape(mask.shape)

        # Define the kernel for dilation
        kernel = np.ones((3, 3), np.uint8)

        pred = cv2.dilate(pred, kernel, iterations=5)
        pred = cv2.erode(pred, kernel, iterations=3)

        plt.figure(figsize=(6, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(mask, cmap='gray')
        plt.title("True Mask")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(pred, cmap='gray')
        plt.title("Predicted Mask")
        plt.axis('off')

        if val == False:
            plt.savefig(f"/content/visualisation/train_{i}.png")
        else:
            plt.savefig(f"/content/visualisation/val_{i}.png")


def train(args, predictor):
    data_path = args.data_path
    assert os.path.exists(data_path), 'data path does not exist!'

    num_image = args.k

    fnames = os.listdir(os.path.join(data_path, 'images'))
    # get k random indices from fnames
    random.shuffle(fnames)
    fnames = fnames[:num_image]

    # Split file names into training and validation sets
    train_fnames, val_fnames = train_test_split(fnames, test_size=0.6, random_state=42)

    # image augmentation and embedding processing
    num_augmentations = int(args.augmentation_num)  # Number of augmented versions to create per image

    def process_images(file_names, augment_data=True):
        image_embeddings = []
        labels = []

        def process_and_store(img, msk):
            # Resize and process the mask and image
            resized_mask = cv2.resize(msk, dsize=(64, 64), interpolation=cv2.INTER_NEAREST)
            resized_img = cv2.resize(img, dsize=(1024, 1024), interpolation=cv2.INTER_NEAREST)

            # Process the image to create an embedding
            img_emb = get_embedding(resized_img, predictor)
            img_emb = img_emb.cpu().numpy().transpose((2, 0, 3, 1)).reshape((256, 64, 64))
            image_embeddings.append(img_emb)
            labels.append(resized_mask)

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

        return image_embeddings, labels

    # Process training images with augmentation
    train_embeddings, train_labels = process_images(train_fnames, augment_data=True)

    # Process validation images without augmentation
    val_embeddings, val_labels = process_images(val_fnames, augment_data=False)

    # Convert to tensors
    train_embeddings_tensor = torch.stack([torch.Tensor(e) for e in train_embeddings])
    train_labels_tensor = torch.stack([torch.Tensor(l) for l in train_labels])
    val_embeddings_tensor = torch.stack([torch.Tensor(e) for e in val_embeddings])
    val_labels_tensor = torch.stack([torch.Tensor(l) for l in val_labels])

    # Use the same function as defined for Random Forest
    train_embeddings_flat, train_labels_flat = create_dataset_for_SVM(train_embeddings_tensor.numpy(),
                                                                     train_labels_tensor.numpy())
    val_embeddings_flat, val_labels_flat = create_dataset_for_SVM(val_embeddings_tensor.numpy(),
                                                                 val_labels_tensor.numpy())

    svm_model = SVC(kernel='linear')  # Or any other kernel
    svm_model.fit(train_embeddings_flat, train_labels_flat)

    # Predict on the validation set
    predicted_masks_svm = predict_and_reshape(svm_model, val_embeddings_flat, (len(val_embeddings_tensor), 64, 64))

    # Define the kernel for dilation
    kernel = np.ones((3, 3), np.uint8)

    predicted_masks_svm = cv2.dilate(predicted_masks_svm, kernel, iterations=5)
    predicted_masks_svm = cv2.erode(predicted_masks_svm, kernel, iterations=3)

    # Evaluate the SVM model
    accuracy_svm = accuracy_score(val_labels_flat, predicted_masks_svm.reshape(-1))
    print(f'SVM Accuracy: {accuracy_svm}')
    print(classification_report(val_labels_flat, predicted_masks_svm.reshape(-1)))

    # # Train a logistic regression model
    # logistic_regression_model = LogisticRegression(max_iter = 10000)
    # logistic_regression_model.fit(train_embeddings_flat, train_labels_flat)
    #
    # # Predict on the validation set
    # predicted_masks_logistic = logistic_regression_model.predict(val_embeddings_flat)
    #
    # # Apply thresholding (e.g., 0.5) to get binary predictions
    # predicted_masks_binary = (predicted_masks_logistic > args.threshold).astype(np.uint8).reshape(len(val_embeddings), 64, 64)

    # Dice Scores
    svm_dice_val = dice_coeff(torch.Tensor(predicted_masks_svm), torch.Tensor(val_labels))
    print('SVM Dice: ', svm_dice_val)
    # log_dice_val = dice_coeff(torch.Tensor(predicted_masks_binary),torch.Tensor(val_labels))
    # print('Logsitic Regression Dice: ', svm_dice_val)

    # # Evaluate the Logistic regression model
    # accuracy_svm = accuracy_score(val_labels_flat, predicted_masks_binary.reshape(-1))
    # print(f'Logistic Regression Accuracy: {accuracy_svm}')
    # print(classification_report(val_labels_flat, predicted_masks_svm.reshape(-1)))

    # # Visualize Logistic regression predictions on the training dataset
    # print("Training Predictions with SVM:")
    # visualize_predictions(train_embeddings, train_labels, logistic_regression_model, val=False)

    # Visualize SVM predictions on the validation dataset
    print("Validation Predictions with SVM:")
    visualize_predictions(val_embeddings, val_labels, svm_model, val=True)

    return svm_model


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