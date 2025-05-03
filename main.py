import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2
from skimage import color
from sklearn.preprocessing import StandardScaler

class ImageKMeansSegmenter:
    def __init__(self, image_path, resize_dim=(200, 200), seed=42):
        self.image_path = image_path
        self.resize_dim = resize_dim
        np.random.seed(seed)
        self.img = self.load_image()
        self.h, self.w, _ = self.img.shape
        
        # Apply Gaussian blur to reduce noise
        self.img_blurred = cv2.GaussianBlur(self.img, (5, 5), 0)
        
        # Convert to Lab color space (expects float [0,1])
        lab_img = color.rgb2lab(self.img_blurred / 255.0)
        
        # Flatten pixels (shape: [height*width, 3])
        self.flat_pixels = lab_img.reshape(-1, 3)
        
        # Normalize Lab pixels using StandardScaler
        self.scaler = StandardScaler()
        self.flat_pixels = self.scaler.fit_transform(self.flat_pixels)

    def load_image(self):
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"Image not found at: {self.image_path}")
        img = Image.open(self.image_path).convert('RGB')
        img = img.resize(self.resize_dim)
        return np.array(img)

    def initialize_centroids_kmeans_pp(self, X, k):
        n_samples = X.shape[0]
        centroids = np.empty((k, X.shape[1]), dtype=X.dtype)

        centroid_idx = np.random.choice(n_samples)
        centroids[0] = X[centroid_idx]
        closest_dist_sq = np.full(n_samples, np.inf)

        for c_id in range(1, k):
            dist_sq = np.sum((X - centroids[c_id - 1]) ** 2, axis=1)
            closest_dist_sq = np.minimum(closest_dist_sq, dist_sq)
            prob = closest_dist_sq / closest_dist_sq.sum()
            next_centroid_idx = np.random.choice(n_samples, p=prob)
            centroids[c_id] = X[next_centroid_idx]

        return centroids

    def assign_clusters(self, X, centroids):
        distances = np.sqrt(((X - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        return np.argmin(distances, axis=0)

    def update_centroids(self, X, labels, k):
        new_centroids = np.array([
            X[labels == i].mean(axis=0) if np.any(labels == i) else X[np.random.choice(len(X))]
            for i in range(k)
        ])
        return new_centroids

    def kmeans(self, X, k, max_iters=100):
        centroids = self.initialize_centroids_kmeans_pp(X, k)
        for _ in range(max_iters):
            labels = self.assign_clusters(X, centroids)
            new_centroids = self.update_centroids(X, labels, k)
            if np.allclose(centroids, new_centroids, atol=1e-4):
                break
            centroids = new_centroids
        inertia = np.sum((X - centroids[labels]) ** 2)
        return centroids, labels, inertia

    def extract_colored_number(self, labels, k):
        """
        Highlights the smallest cluster in the image with fixed green color.
        """
        counts = np.bincount(labels, minlength=k)
        number_cluster = np.argmin(counts)

        result_img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        mask = labels.reshape(self.h, self.w)

        # Always color green to highlight selected cluster
        result_img[mask == number_cluster] = [0, 255, 0]  # green color

        return result_img

    def process_for_k_values(self, k_values):
        results = []
        inertias = []
        for k in k_values:
            centroids, labels, inertia = self.kmeans(self.flat_pixels, k)
            result_img = self.extract_colored_number(labels, k)
            results.append((k, result_img))
            inertias.append(inertia)
        return results, inertias

    @staticmethod
    def find_elbow_point(inertias):
        diffs = np.diff(inertias)
        ddiffs = np.diff(diffs)
        elbow_idx = np.argmin(ddiffs) + 1
        return elbow_idx

    def plot_inertia(self, k_values, inertias):
        plt.figure(figsize=(6, 4))
        plt.plot(k_values, inertias, marker='o', linestyle='--', color='b')
        plt.xlabel("K value")
        plt.ylabel("Inertia")
        plt.title("Elbow Method: Inertia vs K")
        plt.grid(True)
        plt.show()

    def plot_segmented_image(self, segmented_img, k):
        plt.figure(figsize=(6, 6))
        plt.imshow(segmented_img)
        plt.title(f"Extracted Number Using K = {k}")
        plt.axis('off')
        plt.show()


if __name__ == "__main__":
    image_path = r"D:\6.jpg"  # change as needed
    segmenter = ImageKMeansSegmenter(image_path)

    k_values = list(range(2, 8))

    results, inertias = segmenter.process_for_k_values(k_values)

    segmenter.plot_inertia(k_values, inertias)

    elbow_idx = segmenter.find_elbow_point(inertias)
    best_k = k_values[elbow_idx]
    print(f"Best K detected by elbow method: {best_k}")

    # Find result for best K and plot
    best_result_img = None
    for k, segmented_img in results:
        if k == best_k:
            best_result_img = segmented_img
            break

    segmenter.plot_segmented_image(best_result_img, best_k)
