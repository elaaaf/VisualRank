from FeatureExtractor import FeatureExtractor
from datasets import load_dataset
import numpy as np


def main():
    print("Loading dataset...")
    ds = load_dataset("microsoft/cats_vs_dogs")
    extractor = FeatureExtractor(model_name='resnet50', use_gpu=True)
    print("Extracting features...")
    feature_matrix = extractor.extract_features(ds)
    similarity_matrix = extractor.compute_similarity_matrix(feature_matrix)
                                                            
    # Save results
    print("Saving results...")
    np.save('feature_matrix.npy', feature_matrix)
    np.save('similarity_matrix.npy', similarity_matrix)
    print(f"Feature matrix shape: {feature_matrix.shape}")
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    

if __name__=='__main__':
    main()