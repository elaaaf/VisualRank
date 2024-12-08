from FeatureExtractor import FeatureExtractor
from datasets import load_dataset
import numpy as np
from visual_rank import VisualRank
from kmeans_baseline import KMeansBaseline

def main():

    print("Loading dataset...")
    ds = load_dataset("microsoft/cats_vs_dogs",  split="train[:1%]") # minimize dataset for development purposes

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

    print("Applying k-means baseline...")
    kmeans_baseline = KMeansBaseline(n_clusters=5, use_pca=True)
    cluster_labels = kmeans_baseline.cluster(feature_matrix)

    print("Visualizing k-means clusters...")
    kmeans_baseline.visualize_clusters(cluster_labels, ds)
    

    print("Calculating Visual Rank...")
    visual_rank_obj = VisualRank(similarity_matrix)
    visual_rank_scores = visual_rank_obj.calculate_visual_rank()
    np.save('visual_rank_scores.npy', visual_rank_scores)
    print(f"Visual rank scores shape: {len(visual_rank_scores)} (one score per image)")
     # Visualize top-ranked images
    visual_rank_obj.visualize_top_ranked_images(np.array(visual_rank_scores), feature_matrix, similarity_matrix, ds)

if __name__=='__main__':
    main()