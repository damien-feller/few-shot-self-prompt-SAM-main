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
import imgaug.augmenters as iaa
import imgaug as ia


# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class CustomDataset(Dataset):
    def __init__(self, embeddings, labels, original_images, mean=None, std=None):
        self.embeddings = embeddings
        self.labels = labels
        self.original_images = original_images  # Store the original images
        self.mean = mean
        self.std = std

        if self.mean is not None and self.std is not None:
            self.normalize_embeddings()

    def normalize_embeddings(self):
        self.embeddings = (self.embeddings - self.mean) / self.std

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label = self.labels[idx]
        original_image = self.original_images[idx]  # Retrieve the original image
        return embedding, label, original_image


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU => Dropout) * 2"""

    def __init__(self, in_channels, out_channels, dropout_prob=0.5):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),  # Dropout after activation
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob)  # Dropout after activation
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, concat_channels=None):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        if concat_channels is not None:
            self.conv = DoubleConv(in_channels // 2 + concat_channels, out_channels)
        else:
            self.conv = DoubleConv(in_channels // 2, out_channels)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        if x2 is not None:
            # Code for padding and concatenation
            diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
            diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        # Initial double conv to reduce channels from 256 to 64
        self.inc = DoubleConv(n_channels, 64)
        # 64x64x64
        self.down1 = Down(64, 128)
        # 128x32x32
        self.down2 = Down(128, 256)
        # 256x16x16
        self.down3 = Down(256, 512)
        # 512x8x8
        # Upscaling back to 256 channels and 16x16
        self.up1 = Up(512, 256, 256)
        # 256x16x16
        # Upscaling back to 128 channels and 32x32
        self.up2 = Up(256, 128, 128)
        # 128x32x32
        # Upscaling back to 64 channels and 64x64
        self.up3 = Up(128, 64, 64)
        # Output layer to get the required number of classes
        self.outc = OutConv(64, n_classes)
        # Final upsampling to 128x128
        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        # Initial double conv
        x1 = self.inc(x)
        # Downscale
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # Upscale + skip connection from the last downscale layer
        x = self.up1(x4, x3)
        # Further upscale + skip connection from the second to last downscale layer
        x = self.up2(x, x2)
        # Final upscale + skip connection from the initial double conv layer
        x = self.up3(x, x1)
        # Output convolution
        logits = self.outc(x)
        # Final upsampling to 128x128
        #logits = self.final_up(logits)
        return logits


def get_embedding(img, predictor):
    predictor.set_image(img)
    img_emb = predictor.get_image_embedding()
    return img_emb

def visualize_predictions(dataset, model, num_samples=5, val=False, threshold=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    indices = np.random.choice(range(len(dataset)), num_samples, replace=False)
    for i in indices:
        embedding, mask, original_image = dataset[i]  # Now includes the original image
        embedding = embedding.unsqueeze(0).to(device)
        pred = model(embedding)
        pred = torch.sigmoid(pred)
        pred = (pred > threshold).float()

        plt.figure(figsize=(12, 3))

        plt.subplot(1, 3, 1)
        plt.imshow(original_image.squeeze(), cmap='gray')  # Display the original image
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(mask.squeeze(), cmap='gray')
        plt.title("True Mask")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(pred.cpu().squeeze(), cmap='gray')
        plt.title("Predicted Mask")
        plt.axis('off')

        if val == False:
            plt.savefig(f"/content/visualisation/train_{i}.png")
        else:
            plt.savefig(f"/content/visualisation/val_{i}.png")
        plt.close()

def visualize_samples(dataset):
    for i in range(len(dataset)):
        embedding, mask, original_image = dataset[i]  # Now includes the original image
        plt.figure()
        plt.imshow(original_image.squeeze(), cmap='gray')  # Display the original image
        plt.title("Original Image")
        plt.axis('off')

        plt.savefig(f"/content/visualisation/image_{i}.png")
        plt.close()


def plot_losses(train_losses, val_losses, train_dice, val_dice):
    try:
        # Assuming train_losses and val_losses are lists containing loss values for each epoch
        epochs = range(1, len(train_losses) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, label='Training Loss')
        plt.plot(epochs, val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"/content/visualisation/LOSSES.png")
        plt.show()
    except Exception as e:
        print(f"Error occurred while saving the plot: {e}")

    try:
        # Assuming train_losses and val_losses are lists containing loss values for each epoch
        epochs = range(1, len(train_dice) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_dice, label='Training Dice')
        plt.plot(epochs, val_dice, label='Validation Dice')
        plt.title('Training and Validation Dice Score')
        plt.xlabel('Epochs')
        plt.ylabel('Dice Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"/content/visualisation/Dice Score.png")
        plt.show()
    except Exception as e:
        print(f"Error occurred while saving the plot: {e}")

def augment(image, mask):
    # Define an augmentation pipeline
    transform = A.Compose([
        A.Rotate(limit=45, p=0.5),  # Rotation
        A.RandomScale(scale_limit=0.2, p=0.5),  # Scaling
        A.GaussNoise(var_limit=(6, 27), p=0.5),  # Gaussian Noise
        A.GaussianBlur(blur_limit=(3, 9), p=0.5),  # Gaussian Blur
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),  # Brightness & Contrast
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


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, true_masks):
        preds = torch.sigmoid(logits)
        preds_flat = preds.view(-1)
        true_masks_flat = true_masks.view(-1)

        intersection = (preds_flat * true_masks_flat).sum()
        dice_score = (2. * intersection + self.smooth) / (preds_flat.sum() + true_masks_flat.sum() + self.smooth)
        return 1 - dice_score

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=1.5):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, true):
        probs = torch.sigmoid(logits)
        true = true.float()
        true_pos = torch.sum(probs * true, dim=(2, 3))
        false_neg = torch.sum((1 - probs) * true, dim=(2, 3))
        false_pos = torch.sum(probs * (1 - true), dim=(2, 3))
        tversky = (true_pos + self.smooth) / (true_pos + self.alpha * false_neg + self.beta * false_pos + self.smooth)
        return 1 - torch.mean(tversky)

def train(args, predictor):
    data_path = args.data_path
    assert os.path.exists(data_path), 'data path does not exist!'

    num_image = args.k

    fnames = os.listdir(os.path.join(data_path, 'images'))
    # get k random indices from fnames
    random.shuffle(fnames)
    fnames = fnames[:num_image]

    # Split file names into training and validation sets
    train_fnames, val_fnames = train_test_split(fnames, test_size=0.2, random_state=42)

    # image augmentation and embedding processing
    num_augmentations = int(args.augmentation_num)  # Number of augmented versions to create per image

    def process_images(file_names, augment_data=True):
        image_embeddings = []
        labels = []
        original_images = []

        def process_and_store(img, msk):
            # Resize and process the mask and image
            resized_mask = cv2.resize(msk, dsize=(64, 64), interpolation=cv2.INTER_NEAREST)
            resized_img = cv2.resize(img, dsize=(1024, 1024), interpolation=cv2.INTER_NEAREST)
            original_images.append(resized_img)

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

        return image_embeddings, labels, original_images

    # Process training images with augmentation
    train_embeddings, train_labels, train_images = process_images(train_fnames, augment_data=True)

    # Process validation images without augmentation
    val_embeddings, val_labels, val_images = process_images(val_fnames, augment_data=False)

    # Convert to tensors
    train_embeddings_tensor = torch.stack([torch.Tensor(e) for e in train_embeddings])
    train_labels_tensor = torch.stack([torch.Tensor(l) for l in train_labels])
    val_embeddings_tensor = torch.stack([torch.Tensor(e) for e in val_embeddings])
    val_labels_tensor = torch.stack([torch.Tensor(l) for l in val_labels])

    # Calculate mean and std
    mean = train_embeddings_tensor.mean()
    std = train_embeddings_tensor.std()

    # Normalize datasets
    train_dataset = CustomDataset(train_embeddings_tensor, train_labels_tensor, train_images, mean, std)
    val_dataset = CustomDataset(val_embeddings_tensor, val_labels_tensor, val_images, mean, std)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Instantiate the model and move it to the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Instantiate the model
    model = UNet(n_channels=256, n_classes=1).to(device)

    # Loss and optimizer functions
    #criterion = TverskyLoss(alpha=0.5, beta=0.5)
    criterion = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=25, verbose=True)

    train_losses = []
    val_losses = []
    train_dice_scores = []
    val_dice_scores = []

    #visualize_samples(train_dataset)
    #visualize_samples(val_dataset)

    #training cycle
    for epoch in range(args.epochs):
        # Training phase
        train_loss = 0.0
        val_loss = 0.0
        train_dice = 0.0
        val_dice = 0.0

        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Ensure the label is a floating-point tensor
            labels = labels.float().unsqueeze(1)

            # Forward pass (model outputs logits)
            logits = model(images)

            # Compute the loss
            loss = criterion(logits, labels)

            # Compute dice score
            preds = torch.sigmoid(logits)
            preds = (preds > args.threshold).float()
            dice_score = dice_coeff(preds, labels)
            train_dice += dice_score.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        #dice score
        avg_train_dice = train_dice / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(device), labels.to(device)
                labels = labels.float().unsqueeze(1)  # Ensure labels have correct shape

                # Forward pass
                logits = model(images)

                # Calculate the loss
                loss = criterion(logits, labels)

                # Compute dice score
                preds = torch.sigmoid(logits)
                preds = (preds > args.threshold).float()
                dice_score = dice_coeff(preds, labels)
                val_dice += dice_score.item()

                # Accumulate the validation loss
                val_loss += loss.item()

        #dice score
        avg_val_dice = val_dice / len(val_loader)
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)


        # Append average loss per epoch
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        train_dice_scores.append(avg_train_dice)
        val_dice_scores.append(avg_val_dice)

        print(f'Epoch [{epoch + 1}/{args.epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, '
              f'Train Dice: {avg_train_dice:.4f}, Val Dice: {avg_val_dice:.4f}')

    # Visualize training predictions
    print("Training Predictions:")
    visualize_predictions(train_dataset, model, val = False)

    # Visualize validation predictions
    print("Validation Predictions:")
    visualize_predictions(val_dataset, model, val = True, threshold=args.threshold)

    plot_losses(train_losses, val_losses, train_dice_scores, val_dice_scores)

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
    parser.add_argument('--augmentation', type=bool, default=True, help='toggle augmenation')
    args = parser.parse_args()

    # set random seed
    random.seed(42)
    
    # register the SAM model
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(args.device)
    global predictor
    predictor = SamPredictor(sam)
    print('SAM model loaded!', '\n')

    model = train(args, predictor)



if __name__ == '__main__':
    main()