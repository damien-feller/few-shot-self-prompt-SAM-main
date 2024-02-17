import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import random
from tqdm import tqdm
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
import argparse
from sklearn.preprocessing import StandardScaler
import umap
import matplotlib.pyplot as plt


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
def process_images(file_names, data_path, predictor, num_augmentations=0):
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

    for fname in tqdm(file_names[0:5]):
        # Read data
        image = cv2.imread(os.path.join(data_path, 'images', fname))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(data_path, 'masks', fname), cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 128, 1, cv2.THRESH_BINARY)
        process_and_store(image, mask)

    return image_embeddings, labels


def visualize_umap(embeddings, labels, n_neighbors=15, min_dist=0.1, n_components=2):
    # Convert list of embeddings and labels to NumPy arrays
    embeddings_array = np.array(embeddings)
    labels_array = np.array(labels)

    # Since embeddings are expected to be 4D (N, C, H, W), ensure they are correctly reshaped
    # Check if embeddings_array is already in the expected shape or needs reshaping
    if embeddings_array.ndim == 4:
        N, C, H, W = embeddings_array.shape
        print(embeddings_array.shape)
        embeddings_flat = embeddings_array.reshape(C, -1)
        print(embeddings_flat.shape)
    else:
        # Handle case where embeddings might not be in the expected format
        raise ValueError("Embeddings array is not in the expected shape of (N, C, H, W)")

    labels_flat = labels_array.flatten()
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components)
    embedding = reducer.fit_transform(np.transpose(embeddings_flat))
    print(embedding.shape)

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels_flat, cmap='Spectral', s=5)
    plt.colorbar(scatter, spacing='proportional', label='Class')
    plt.title('UMAP projection of the dataset')
    plt.show()
    plt.savefig(f"/content/UMAP.png")



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

    # Initialize SAM predictor
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(args.device)
    predictor = SamPredictor(sam)

    # Load dataset
    fnames = os.listdir(os.path.join(args.data_path, 'images'))
    embeddings, labels = process_images(fnames, args.data_path, predictor)


    # UMAP visualization
    visualize_umap(embeddings, labels)


if __name__ == '__main__':
    main()