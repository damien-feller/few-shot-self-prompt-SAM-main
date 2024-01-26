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

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# class CustomDataset(Dataset):
#     def __init__(self, embeddings, labels):
#         self.embeddings = embeddings
#         self.labels = labels
#
#     def __len__(self):
#         return len(self.embeddings)
#
#     def __getitem__(self, idx):
#         image = self.embeddings[idx]
#         label = self.labels[idx]
#         return image, label

class CustomDataset(Dataset):
    def __init__(self, image_paths, mask_paths, predictor, is_train=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.predictor = predictor
        self.is_train = is_train

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and mask
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 128, 1, cv2.THRESH_BINARY)

        # Perform augmentation if it's a training dataset
        if self.is_train:
            augmented_image, augmented_mask = augment(image, mask)
        else:
            augmented_image, augmented_mask = image, mask

        # Resize and process the augmented mask
        resized_mask = cv2.resize(augmented_mask, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)

        # Compute the embedding for the augmented image
        img_emb = get_embedding(augmented_image, self.predictor)
        img_emb = img_emb.cpu().numpy().transpose((2, 0, 3, 1)).reshape((256, 64, 64))

        return torch.Tensor(img_emb), torch.Tensor(resized_mask)

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
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# class UNet(nn.Module):
#     def __init__(self, n_channels, n_classes):
#         super(UNet, self).__init__()
#         self.inc = DoubleConv(n_channels, 64)   #
#         self.down1 = Down(64, 128)
#         self.down2 = Down(128, 256)
#         self.up1 = Up(256, 128)
#         self.up2 = Up(128, 64)
#         self.up3 = Up(64, 32)  # Additional upsampling layer
#         self.up4 = Up(32, 16)  # Additional upsampling layer
#         self.outc = OutConv(16, n_classes)  # Adjust the number of output channels to match n_classes
#
#     def forward(self, x):
#         # Downsampling path
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#
#         # Upsampling path with skip connections
#         x = self.up1(x3, x2)
#         x = self.up2(x, x1)
#
#         # If you don't have additional layers for skip connections in up3 and up4,
#         # you might consider not using them or redesigning your architecture.
#         # As an example, just passing through additional convolutions:
#         x = self.up3.conv(x)  # Modified to use only conv part of the Up module
#         x = self.up4.conv(x)  # Modified to use only conv part of the Up module
#
#         logits = self.outc(x)
#         return logits

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, output_size=(256, 256)):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        # Removed the second down layer
        self.up1 = Up(128, 64)  # Changed input channels to match output of down1
        # Direct Upsampling Layer
        self.upsample = nn.Upsample(size=output_size, mode='bilinear', align_corners=True)
        self.outc = OutConv(64, n_classes)  # No change

    def forward(self, x):
        # Downsampling path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        # Removed the operation involving down2

        # Upsampling path with skip connections
        # The output of down1 is now used as the input to the first upsampling layer
        x = self.up1(x2, x1)

        # Direct Upsampling to the final output size
        x = self.upsample(x)

        logits = self.outc(x)
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
        image, mask = dataset[i]
        image = image.unsqueeze(0).to(device)
        pred = model(image)
        pred = torch.sigmoid(pred)
        pred = (pred > threshold).float()

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
        A.GaussNoise(var_limit=(10, 50), p=0.5),  # Gaussian Noise
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),  # Gaussian Blur
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),  # Brightness & Contrast
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),  # Gamma Augmentation
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

def dice_loss(pred, target):
    smooth = 1.
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1 - (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def train(args, predictor):
    data_path = args.data_path
    assert os.path.exists(data_path), 'data path does not exist!'

    # Load the arguments (assuming args.k exists and is the number of images to load)
    num_images = args.k
    image_dir = os.path.join(data_path, 'images')
    mask_dir = os.path.join(data_path, 'masks')

    # Get all file names
    all_image_files = sorted(os.listdir(image_dir))  # Sort to ensure consistent ordering
    all_mask_files = sorted(os.listdir(mask_dir))

    # Select only the first 'num_images' files
    selected_image_files = all_image_files[:num_images]
    selected_mask_files = all_mask_files[:num_images]

    # Create full paths
    image_paths = [os.path.join(image_dir, fname) for fname in selected_image_files]
    mask_paths = [os.path.join(mask_dir, fname) for fname in selected_mask_files]

    #dataset intialisation
    train_image_paths, val_image_paths, train_mask_paths, val_mask_paths = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42)

    train_dataset = CustomDataset(train_image_paths, train_mask_paths, predictor, is_train=True)
    val_dataset = CustomDataset(val_image_paths, val_mask_paths, predictor, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Instantiate the model and move it to the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Instantiate the model
    model = UNet(n_channels=256, n_classes=1).to(device)

    # Loss and optimizer functions
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    #training cycle
    train_losses = []
    val_losses = []
    train_dice_scores = []
    val_dice_scores = []

    for epoch in range(args.epochs):
        model.train()
        # Training phase
        train_loss = 0.0
        val_loss = 0.0
        train_dice = 0.0
        val_dice = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Ensure the label is a floating-point tensor
            labels = labels.float().unsqueeze(1)

            # Forward pass (model outputs logits)
            logits = model(images)
            preds = torch.sigmoid(logits)
            preds = (preds > args.threshold).float()

            # Compute the loss
            loss = dice_loss(preds, labels)

            # Compute dice score
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
            for images, labels in val_loader:
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

def test_visualize(args, model, predictor):
    data_path = args.data_path
        
    num_image = args.k
    fnames = os.listdir(os.path.join(data_path, 'images'))
    random.shuffle(fnames)
    fnames = fnames[num_image:]
    num_visualize = args.visualize_num
    
    dice_linear = []
    dice1 = []
    dice2 = []
    dice3 = []
    i = 0

    for fname in tqdm(fnames[:num_visualize]):
        # read data
        image = cv2.imread(os.path.join(data_path, 'images', fname))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(data_path, 'masks', fname))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 128, 1, cv2.THRESH_BINARY)
        H, W, _ = image.shape

        # get the image embedding for CNN
        img_emb = get_embedding(image, predictor)
        img_emb = img_emb.cpu().numpy().transpose((2, 0, 3, 1)).reshape((256, 64, 64))

        # CNN prediction
        img_emb_tensor = torch.Tensor(img_emb).unsqueeze(0).to(args.device)  # Add batch dimension and send to device
        y_pred = model(img_emb_tensor)
        y_pred = torch.argmax(y_pred, dim=1).squeeze(
            0).cpu().numpy()  # Assuming the model outputs logits for each class
        mask_pred_l = cv2.resize(y_pred, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)

        # use distance transform to find a point inside the mask
        fg_point = get_max_dist_point(mask_pred_l)
        
        # set the image to sam
        predictor.set_image(image)
        
        # prompt the sam with the point
        input_point = np.array([[fg_point[0], fg_point[1]]])
        input_label = np.array([1])
        masks_pred_sam_prompted1, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=None,
            multimask_output=False,
        )
        
        # prompt the sam with the bounding box
        y_indices, x_indices = np.where(mask_pred_l > 0)
        if np.all(mask_pred_l == 0):
            bbox = np.array([0, 0, H, W])
        else:
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            H, W = mask_pred_l.shape
            x_min = max(0, x_min - np.random.randint(0, 20))
            x_max = min(W, x_max + np.random.randint(0, 20))
            y_min = max(0, y_min - np.random.randint(0, 20))
            y_max = min(H, y_max + np.random.randint(0, 20))
            bbox = np.array([x_min, y_min, x_max, y_max])
        masks_pred_sam_prompted2, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=bbox[None, :],
            multimask_output=False,)
            
        # prompt the sam with both the point and bounding box
        masks_pred_sam_prompted3, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=bbox[None, :],
            multimask_output=False,)
        
        dice_l = dice_coef(mask, mask_pred_l)
        dice_p = dice_coef(mask, masks_pred_sam_prompted1[0])
        dice_b = dice_coef(mask, masks_pred_sam_prompted2[0])
        dice_i = dice_coef(mask, masks_pred_sam_prompted3[0])
        dice_linear.append(dice_l)
        dice1.append(dice_p)
        dice2.append(dice_b)
        dice3.append(dice_i)

        # plot the results
        fig, ax = plt.subplots(1, 5, figsize=(15, 10))
        ax[0].set_title('Ground Truth')
        ax[0].imshow(mask)
        ax[1].set_title('Linear + e&d')
        ax[1].plot(fg_point[0], fg_point[1], 'r.')
        ax[1].imshow(mask_pred_l)
        ax[2].set_title('Point')
        ax[2].plot(fg_point[0], fg_point[1], 'r.')
        ax[2].imshow(masks_pred_sam_prompted1[0]) 
        ax[3].set_title('Box')
        show_box(bbox, ax[3])
        ax[3].imshow(masks_pred_sam_prompted2[0])
        ax[4].set_title('Point + Box')
        ax[4].plot(fg_point[0], fg_point[1], 'r.')
        show_box(bbox, ax[4])
        ax[4].imshow(masks_pred_sam_prompted3[0])
        [axi.set_axis_off() for axi in ax.ravel()]
        
        
        if os.path.exists(args.save_path) == False:
            os.mkdir(args.save_path)
        plt.savefig(os.path.join(args.save_path, fname.split('.')[0]+str(i)))
    
    mdice0 = round(sum(dice_linear)/float(len(dice_linear)), 5)
    mdice1 = round(sum(dice1)/float(len(dice1)), 5)
    mdice2 = round(sum(dice2)/float(len(dice2)), 5)
    mdice3 = round(sum(dice3)/float(len(dice3)), 5)
    
    print('For the first {} images: '.format(num_visualize))
    print('mdice(linear classifier: )', mdice0)
    print('mDice(point prompts): ', mdice1)
    print('mDice(bbox prompts): ', mdice2)
    print('mDice(points and boxes): ', mdice3)

        
        
def test(args, predictor):
    data_path = args.data_path
    images = []
    masks = []
    fnames = os.listdir(os.path.join(data_path, 'images'))
    print(f'loading images from {data_path}...')
    for fname in tqdm(fnames):
        # read data
        image = cv2.imread(os.path.join(data_path, 'images', fname))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(data_path, 'masks', fname))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 128, 1, cv2.THRESH_BINARY)
        images.append(image)
        masks.append(mask)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, text_index in kf.split(images):
        train_images = [images[i] for i in train_index]
        train_masks = [masks[i] for i in train_index]
        test_images = [images[i] for i in text_index]
        test_masks = [masks[i] for i in text_index]
        
        # train the linear classifier
        k = args.k
        random_indices = random.sample(range(len(train_images)), k)
        image_embeddings = []
        labels = []
        for idx in random_indices:
            image = train_images[idx]
            mask = train_masks[idx]
            resized_mask = cv2.resize(mask, dsize=(64, 64), interpolation=cv2.INTER_NEAREST)

            img_emb = get_embedding(image)
            img_emb = img_emb.cpu().numpy().transpose((2, 3, 1, 0)).reshape((64, 64, 256))
            image_embeddings.append(img_emb)
            labels.append(resized_mask)

        # Create tensors from image embeddings and labels
        image_embeddings_tensor = torch.stack([torch.Tensor(e) for e in image_embeddings])
        labels_tensor = torch.stack([torch.Tensor(l) for l in labels])

        # Create a CNN model to train on image embeddings and labels
        train_dataset = CustomDataset(image_embeddings_tensor, labels_tensor)
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

        # Instantiate the U-Net model and move it to the device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = UNet(n_channels=256, n_classes=1).to(device)  # Adjust n_channels and n_classes as needed

        # For binary segmentation, use BCEWithLogitsLoss
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Training loop

        for epoch in range(args.epochs):
            model.train()  # Set the model to training mode
            total_loss = 0.0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                # Ensure the label is a floating-point tensor
                labels = labels.float()

                # Forward pass
                logits = model(images)

                # Compute the loss
                labels = labels.unsqueeze(1)
                loss = criterion(logits, labels)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            # Print the average loss for this epoch
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Avg Loss: {avg_loss:.4f}')

        with torch.no_grad():
            for i in range(5):  # Check 5 random samples
                idx = np.random.randint(0, len(train_dataset))
                image, true_mask = train_dataset[idx]
                image = image.unsqueeze(0).to(device)  # Add batch dimension and transfer to device
                pred_mask = model(image)
                pred_mask = torch.sigmoid(pred_mask)  # Apply sigmoid to get probabilities
                pred_mask = (pred_mask > 0.5).float()  # Threshold the probabilities to get binary mask

                plt.figure(figsize=(10, 4))
                plt.subplot(1, 3, 1)
                plt.imshow(image.cpu().squeeze(), cmap='gray')
                plt.title("Input Image")
                plt.axis('off')

                plt.subplot(1, 3, 2)
                plt.imshow(true_mask.squeeze(), cmap='gray')
                plt.title("True Mask")
                plt.axis('off')

                plt.subplot(1, 3, 3)
                plt.imshow(pred_mask.cpu().squeeze(), cmap='gray')
                plt.title("Predicted Mask")
                plt.axis('off')
                plt.savefig(f"/content/visualisation/plot_{i}.png")

        # test
        dice_linear=[]
        dice1=[]
        dice2=[]
        dice3=[]
        for idx in range(len(test_images)):
            image = test_images[idx]
            mask = test_masks[idx]
            H, W, _ = image.shape

            # get the image embedding for CNN
            img_emb = get_embedding(image, predictor)
            img_emb = img_emb.cpu().numpy().transpose((2, 3, 1, 0)).reshape((64, 64, 256))

            # CNN prediction
            img_emb_tensor = torch.Tensor(img_emb).unsqueeze(0).to(args.device)
            y_pred = model(img_emb_tensor)
            y_pred = torch.argmax(y_pred, dim=1).squeeze(0).cpu().numpy()
            mask_pred_l = cv2.resize(y_pred, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)

            # use distance transform to find a point inside the mask
            fg_point = get_max_dist_point(mask_pred_l)

            # set the image to sam
            predictor.set_image(image)

            # prompt sam with the point
            input_point = np.array([[fg_point[0], fg_point[1]]])
            input_label = np.array([1])
            masks_pred_sam_prompted1, _, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=None,
            multimask_output=False,)

            # prompt sam with the bbox
            y_indices, x_indices = np.where(mask_pred_l > 0)
            if np.all(mask_pred_l==0):
                bbox = np.array([0 ,0, H, W])
            else:
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                H, W = mask_pred_l.shape
                x_min = max(0, x_min - np.random.randint(0, 20))
                x_max = min(W, x_max + np.random.randint(0, 20))
                y_min = max(0, y_min - np.random.randint(0, 20))
                y_max = min(H, y_max + np.random.randint(0, 20))
                bbox = np.array([x_min, y_min, x_max, y_max])
                masks_pred_sam_prompted2, _, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=bbox[None, :],
                multimask_output=False,)

                masks_pred_sam_prompted3, _, _,= predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                box=bbox[None, :],
                multimask_output=False,)

                dice_l = dice_coef(mask, mask_pred_l)
                dice_p = dice_coef(mask, masks_pred_sam_prompted1[0])
                dice_b = dice_coef(mask, masks_pred_sam_prompted2[0])
                dice_c = dice_coef(mask, masks_pred_sam_prompted3[0])
                dice_linear.append(dice_l)
                dice1.append(dice_p)
                dice2.append(dice_b)
                dice3.append(dice_c)
                
        mdice0 = round(sum(dice_linear)/float(len(dice_linear)), 5)
        mdice1 = round(sum(dice1)/float(len(dice1)), 5)
        mdice2 = round(sum(dice2)/float(len(dice2)), 5)
        mdice3 = round(sum(dice3)/float(len(dice3)), 5)

        print('mdice(linear classifier: )', mdice0)
        print('mDice(point prompts): ', mdice1)
        print('mDice(bbox prompts): ', mdice2)
        print('mDice(points and boxes): ', mdice3)
        print('\n')

    

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
        test_visualize(args, model, predictor)
    else:
        test(args, predictor)


if __name__ == '__main__':
    main()