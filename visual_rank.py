import numpy as np
import matplotlib.pyplot as plt

class VisualRank:
    """
    This class calculates Visual Rank scores for images based on their feature matrix.
    """
    def __init__(self, similarity_matrix):
        self.similarity_matrix = similarity_matrix

    def calculate_visual_rank(self, alpha=0.85, max_iterations=100, tolerance=1e-6):
        """
        Calculates Visual Rank scores for images based on their similarity matrix.

        Args:
            similarity_matrix: A square matrix where each element (i, j) represents the similarity between images i and j.
            alpha: Damping factor for PageRank (default: 0.85)
            max_iterations: Maximum number of iterations for the power iteration method (default: 100)
            tolerance: Tolerance for convergence (default: 1e-6)

        Returns:
            A list of Visual Rank scores, one for each image.
        """
        num_images = self.similarity_matrix.shape[0]
        pr = np.ones(num_images) / num_images  # Initial PageRank scores

        for _ in range(max_iterations):
            new_pr = alpha * np.dot(self.similarity_matrix, pr) + (1 - alpha) / num_images
            if np.linalg.norm(new_pr - pr) < tolerance:
                break
            pr = new_pr

        return pr.tolist()
    

    def visualize_top_ranked_images(self, visual_rank_scores, feature_matrix, similarity_matrix, dataset, top_k=5):
        # Get indices of top-ranked images based on visual rank scores
        print(dataset)
        top_indices = np.argsort(visual_rank_scores)[::-1][:top_k]

        print(dataset[3]['image'])
        for i in top_indices:
            print(i)
            plt.figure(figsize=(15, 5))
            plt.subplot(1, top_k+1, 1)
            plt.imshow(dataset[int(i)]['image'])
            plt.title(f"Top {i+1} Ranked Image")
            plt.axis('off')

            # Get indices of most similar images to the top-ranked image
            similar_indices = np.argsort(similarity_matrix[i])[::-1][1:top_k+1]

            for j, similar_idx in enumerate(similar_indices):
                plt.subplot(1, top_k+1, j+2)
                plt.imshow(dataset[int(similar_idx)]['image'])
                plt.title(f"Similar Image {j+1}")
                plt.axis('off')

            plt.show()
