import numpy as np
from HFDataset import HFDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm


class FeatureExtractor:
    def __init__(self, model_name='resnet50', use_gpu=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load pre-trained model
        
        if model_name == 'resnet50':
            model = models.resnet50(weights='IMAGENET1K_V2')
            self.model = nn.Sequential(*list(model.children())[:-1])

        elif model_name == 'vgg16':
            model = models.vgg16(weights='IMAGENET1K_V1')
            self.model = nn.Sequential(*list(model.features.children()))
        
        self.model = self.model.to(self.device)
        self.model.eval()


    def extract_features(self, dataset, batch_size=32):

        ds = HFDataset(dataset)

        dataloader = DataLoader(ds,
                                num_workers=0,
                                shuffle=False,
                                pin_memory=True,
                                batch_size=batch_size)
        
        features_list = []

        with torch.no_grad():
            for images, _ in tqdm(dataloader, desc=f"Extracting features from dataset"):

                images = images.to(self.device)
                features = self.model(images)

                features = features.squeeze().cpu().numpy()

                # Handle single image case
                if len(features.shape) == 1:
                    features = features.reshape(1, -1)

                features_list.append(features)

        feature_matrix = np.vstack(features_list)

        return feature_matrix

    def compute_similarity_matrix(self, feature_matrix, metric='cosine_similarity'):

        if metric == 'cosine_similarity':
            # cos(theta) = A*B / (||A||*||B||) = A/||A|| * B/||B||
            norm = np.linalg.norm(feature_matrix, axis=1)[:,np.newaxis]
            eps = 1e-10  # Small constant to avoid division by zero
            normalized_feature_matrix = feature_matrix / (norm + eps)
            return np.dot(normalized_feature_matrix, normalized_feature_matrix.T)
            

        elif metric == 'euclidean_distance':
            #||x - y||² = (x - y)ᵀ(x - y) = ||x||² + ||y||² - 2xᵀy
            x = np.sum(feature_matrix**2, axis=1)[:, np.newaxis]
            y = np.sum(feature_matrix**2, axis=1)
            two_xy = 2 * np.dot(feature_matrix, feature_matrix.T)

            return (x + y - two_xy)
        
        else:
            print("Please choose either 'cosine_similarity' or 'euclidean_distance'.")

