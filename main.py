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
from sklearn.svm import SVC
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
    indices = np.random.choice(range(len(images)), num_samples, replace=False)

    for i in indices:
        image = images[i]
        mask = masks[i]

        # Flatten the image for SVM prediction
        image_flat = image.reshape(-1, image.shape[0])
        pred_flat = model.predict(image_flat)
        # Reshape the prediction to the original mask shape
        pred = pred_flat.reshape(mask.shape)

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
    train_fnames, val_fnames = train_test_split(fnames, test_size=0.2, random_state=42)

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

    # Evaluate the SVM model
    accuracy_svm = accuracy_score(val_labels_flat, predicted_masks_svm.reshape(-1))
    print(f'SVM Accuracy: {accuracy_svm}')
    print(classification_report(val_labels_flat, predicted_masks_svm.reshape(-1)))

    # Visualize SVM predictions on the training dataset
    print("Training Predictions with SVM:")
    visualize_predictions(train_embeddings, train_labels, svm_model, val=False)

    # Visualize SVM predictions on the validation dataset
    print("Validation Predictions with SVM:")
    visualize_predictions(val_embeddings, val_labels, svm_model, val=True)

    return svm_model

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