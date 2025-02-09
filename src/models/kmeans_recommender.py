import numpy as np
from collections import defaultdict
import time

class KMeansRecommender:
    def __init__(self, n_clusters=100, max_iter=100, batch_size=1000):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.labels = None
        self.centroids = None
    
    def _init_centroids(self, features):
        n_samples, n_features = features.shape
        centroids = np.zeros((self.n_clusters, n_features))
        
        centroids[0] = features[np.random.randint(n_samples)]
        
        for k in range(1, self.n_clusters):

            distances = np.min([
                np.linalg.norm(features - centroid, axis=1) 
                for centroid in centroids[:k]
            ], axis=0)
            
            probs = distances ** 2
            probs /= probs.sum()
            centroids[k] = features[np.random.choice(n_samples, p=probs)]
            
            if k % 10 == 0:
                print(f"Initialized {k}/{self.n_clusters} centroids")
        
        return centroids
    
    def fit(self, features):
        start_time = time.time()
        print("\nFitting K-means Clustering...")
        print(f"Parameters: n_clusters={self.n_clusters}, max_iter={self.max_iter}")
        
        n_samples = features.shape[0]
        
        print("Initializing centroids...")
        self.centroids = self._init_centroids(features)

        for iteration in range(self.max_iter):
            iter_start = time.time()
            old_centroids = self.centroids.copy()
            
            new_centroids = np.zeros_like(self.centroids)
            counts = np.zeros(self.n_clusters)
            self.labels = np.zeros(n_samples, dtype=int)
            
            n_batches = (n_samples - 1) // self.batch_size + 1
            
            for batch_idx in range(n_batches):
                batch_start = batch_idx * self.batch_size
                batch_end = min((batch_idx + 1) * self.batch_size, n_samples)
                batch_features = features[batch_start:batch_end]
                
                distances = np.linalg.norm(
                    batch_features[:, np.newaxis] - self.centroids, 
                    axis=2
                )
                
                batch_labels = np.argmin(distances, axis=1)
                self.labels[batch_start:batch_end] = batch_labels
                
                for k in range(self.n_clusters):
                    mask = (batch_labels == k)
                    if np.any(mask):
                        new_centroids[k] += batch_features[mask].sum(axis=0)
                        counts[k] += mask.sum()
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"Processed {batch_idx + 1}/{n_batches} batches")
            
            mask = counts > 0
            new_centroids[mask] /= counts[mask, np.newaxis]
            new_centroids[~mask] = old_centroids[~mask]
            self.centroids = new_centroids
            
            centroid_shift = np.linalg.norm(self.centroids - old_centroids)
            iter_time = time.time() - iter_start
            print(f"Iteration {iteration + 1}/{self.max_iter}, "
                  f"centroid shift: {centroid_shift:.6f}, "
                  f"time: {iter_time:.2f}s")
            
            if centroid_shift < 1e-4:
                print("Converged!")
                break
        
        cluster_sizes = np.bincount(self.labels)
        print("\nClustering Statistics:")
        print(f"Mean cluster size: {cluster_sizes.mean():.1f}")
        print(f"Min cluster size: {cluster_sizes.min()}")
        print(f"Max cluster size: {cluster_sizes.max()}")
        print(f"Total time: {time.time() - start_time:.2f}s")
        
        return self
    
    def compute_similarity_matrix(self):
        if self.labels is None:
            raise ValueError("Must call fit() before computing similarity matrix")
        
        print("\nComputing similarity matrix...")
        start_time = time.time()
        
        n_samples = len(self.labels)
        similarity_matrix = np.zeros((n_samples, n_samples), dtype=np.float32)
        
        clusters = defaultdict(list)
        for idx, label in enumerate(self.labels):
            clusters[label].append(idx)
        
        for i, (cluster_id, cluster_points) in enumerate(clusters.items()):
            cluster_points = np.array(cluster_points)
            similarity_matrix[cluster_points[:, None], cluster_points] = 1.0
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(clusters)} clusters")
        
        print(f"Similarity matrix computation completed in {time.time() - start_time:.2f}s")
        return similarity_matrix 