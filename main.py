import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

class ImageKMeansSegmenter:
    def __init__(self, filepath, resize_size=(200, 200), seed=42):
        self.filepath = filepath
        self.resize_size = resize_size
        np.random.seed(seed)
        
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Cannot find image at '{filepath}'")
        
        self.image = self._open_and_resize_image()
        self.height, self.width, _ = self.image.shape
        self.data = self.image.reshape(-1, 3).astype(np.float32)

    def _open_and_resize_image(self):
        img = Image.open(self.filepath).convert("RGB")
        img_resized = img.resize(self.resize_size)
        return np.array(img_resized)
    
    def _choose_initial_centers(self, points, num_clusters):
        n_points = points.shape[0]
        centers = np.empty((num_clusters, points.shape[1]), dtype=points.dtype)
        first_center_idx = np.random.choice(n_points)
        centers[0] = points[first_center_idx]

        min_dist_sq = np.full(n_points, np.inf)

        for i in range(1, num_clusters):
            dist_sq = np.sum((points - centers[i-1])**2, axis=1)
            min_dist_sq = np.minimum(min_dist_sq, dist_sq)
            probabilities = min_dist_sq / min_dist_sq.sum()
            next_center_idx = np.random.choice(n_points, p=probabilities)
            centers[i] = points[next_center_idx]

        return centers
    
    def _classify_pixels(self, data, centers):
        distances = np.sqrt(np.sum((data[:, None] - centers)**2, axis=2))
        return np.argmin(distances, axis=1)
    
    def _recompute_centers(self, data, labels, num_clusters):
        new_centers = []
        for cluster_id in range(num_clusters):
            cluster_points = data[labels == cluster_id]
            if len(cluster_points) > 0:
                new_centers.append(cluster_points.mean(axis=0))
            else:
                new_centers.append(data[np.random.choice(len(data))])
        return np.array(new_centers)

    def _run_kmeans(self, data, num_clusters, iterations=100):
        centers = self._choose_initial_centers(data, num_clusters)
        for _ in range(iterations):
            labels = self._classify_pixels(data, centers)
            updated_centers = self._recompute_centers(data, labels, num_clusters)
            if np.allclose(centers, updated_centers, atol=1e-4):
                break
            centers = updated_centers
        total_inertia = np.sum((data - centers[labels]) ** 2)
        return centers, labels, total_inertia
    
    def _highlight_smallest_group_cluster(self, labels, num_clusters):
        pixel_counts = np.bincount(labels, minlength=num_clusters)
        target_cluster = np.argmin(pixel_counts)
        highlighted_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        label_reshaped = labels.reshape(self.height, self.width)
        highlighted_image[label_reshaped == target_cluster] = [0, 255, 0]  # Green highlight
        return highlighted_image
    
    def analyze_image(self, clusters_range=range(2, 8)):
        segmented_images = []
        inertias = []
        for k in clusters_range:
            centers, labels, inertia = self._run_kmeans(self.data, k)
            visual = self._highlight_smallest_group_cluster(labels, k)
            segmented_images.append((k, visual))
            inertias.append(inertia)
        return segmented_images, inertias, list(clusters_range)

    @staticmethod
    def detect_elbow_point(inertia_values):
        first_diff = np.diff(inertia_values)
        second_diff = np.diff(first_diff)
        return np.argmin(second_diff) + 1  # Adjust index for second diff size


if __name__ == "__main__":
    image_path = r"D:\42.jpg"
    segmenter = ImageKMeansSegmenter(image_path)

    k_values = list(range(2, 8))
    results, inertias, _ = segmenter.analyze_image(k_values)

    # Plot inertia curve
    plt.figure(figsize=(8,5))
    plt.plot(k_values, inertias, marker='o')
    plt.xlabel('Number of clusters K')
    plt.ylabel('Inertia (Sum of squared distances)')
    plt.title('Elbow Method For Optimal K')
    plt.grid(True)
    plt.show()

    # Find best K via elbow method
    elbow_idx = ImageKMeansSegmenter.detect_elbow_point(inertias)
    best_k = k_values[elbow_idx]
    print(f"Best K detected by elbow method: {best_k}")

    # Display segmented image for best K
    best_result_img = None
    for k, segmented_img in results:
        if k == best_k:
            best_result_img = segmented_img
            break

    plt.figure(figsize=(6,6))
    plt.imshow(best_result_img)
    plt.title(f'Segmented Image with K={best_k} (Smallest Group Highlighted)')
    plt.axis('off')
    plt.show()
