from FeatureExtractor import FeatureExtractor
from datasets import load_dataset
import numpy as np
from visual_rank import VisualRank
from kmeans_baseline import KMeansBaseline


def compute_precision_recall_f1(predicted_indices, ground_truth_labels, query_label):
    """
    Computes precision, recall, and F1 score for a given query.

    Args:
        predicted_indices: List of indices of predicted images (e.g., ranked list or cluster).
        ground_truth_labels: Array of ground truth labels for all images.
        query_label: The label of the query image.

    Returns:
        precision: Fraction of relevant images in the predicted set.
        recall: Fraction of relevant images retrieved out of all relevant images.
        f1: Harmonic mean of precision and recall.
    """
    relevant_count = np.sum(ground_truth_labels == query_label)  # Total relevant images
    retrieved_relevant_count = np.sum(ground_truth_labels[predicted_indices] == query_label)  # Relevant in predictions

    precision = retrieved_relevant_count / len(predicted_indices) if len(predicted_indices) > 0 else 0
    recall = retrieved_relevant_count / relevant_count if relevant_count > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1

def main():

    print("Loading dataset...")
    ds = load_dataset("microsoft/cats_vs_dogs",  split="train[:100%]") # minimize dataset for development purposes
    ground_truth_labels = np.array([example['labels'] for example in ds])  # 0 for cat, 1 for dog

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

    # Evaluate k-means
    print("Evaluating K-Means Baseline...")
    kmeans_precisions, kmeans_recalls, kmeans_f1s = [], [], []
    for cluster in range(kmeans_baseline.n_clusters):
        cluster_indices = np.where(cluster_labels == cluster)[0]
        if len(cluster_indices) == 0:  # Skip empty clusters
            continue

        # Determine the dominant label in the cluster
        dominant_label = np.bincount(ground_truth_labels[cluster_indices]).argmax()

        # Compute precision, recall, and F1 score
        precision, recall, f1 = compute_precision_recall_f1(cluster_indices, ground_truth_labels, dominant_label)
        kmeans_precisions.append(precision)
        kmeans_recalls.append(recall)
        kmeans_f1s.append(f1)

    print(f"K-Means - Average Precision: {np.mean(kmeans_precisions):.4f}, "
          f"Average Recall: {np.mean(kmeans_recalls):.4f}, "
          f"Average F1 Score: {np.mean(kmeans_f1s):.4f}")
    

    print("Calculating Visual Rank...")
    visual_rank_obj = VisualRank(similarity_matrix)
    visual_rank_scores = visual_rank_obj.calculate_visual_rank()
    np.save('visual_rank_scores.npy', visual_rank_scores)
    print(f"Visual rank scores shape: {len(visual_rank_scores)} (one score per image)")
     # Visualize top-ranked images
    visual_rank_obj.visualize_top_ranked_images(np.array(visual_rank_scores), feature_matrix, similarity_matrix, ds)

    # Evaluate VisualRank
    print("Evaluating Visual Rank...")
    visual_rank_precisions, visual_rank_recalls, visual_rank_f1s = [], [], []
    for i in range(len(ds)):
        # Get top-k indices (excluding the query image itself)
        ranked_indices = np.argsort(similarity_matrix[i])[::-1][1:6]
        query_label = ground_truth_labels[i]

        # Compute precision, recall, and F1 score
        precision, recall, f1 = compute_precision_recall_f1(ranked_indices, ground_truth_labels, query_label)
        visual_rank_precisions.append(precision)
        visual_rank_recalls.append(recall)
        visual_rank_f1s.append(f1)

    print(f"VisualRank - Average Precision: {np.mean(visual_rank_precisions):.4f}, "
          f"Average Recall: {np.mean(visual_rank_recalls):.4f}, "
          f"Average F1 Score: {np.mean(visual_rank_f1s):.4f}")
    
    # Compare results
    print("\nComparison:")
    print(f"VisualRank - Precision: {np.mean(visual_rank_precisions):.4f}, Recall: {np.mean(visual_rank_recalls):.4f}, "
          f"F1 Score: {np.mean(visual_rank_f1s):.4f}")
    print(f"K-Means - Precision: {np.mean(kmeans_precisions):.4f}, Recall: {np.mean(kmeans_recalls):.4f}, "
          f"F1 Score: {np.mean(kmeans_f1s):.4f}")

if __name__=='__main__':
    main()