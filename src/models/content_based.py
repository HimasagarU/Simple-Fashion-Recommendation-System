import numpy as np
import time
import os

class ContentBasedSimilarity:
    def __init__(self, batch_size=1000):
        self.batch_size = batch_size
        self.output_path = 'data/content_similarity.npy'
        
    def _compute_categorical_similarity(self, feature_vector, batch_start, batch_end):

        n_samples = len(feature_vector)
        batch_similarity = np.zeros((batch_end - batch_start, n_samples))
        
        for i in range(batch_start, batch_end):
            batch_similarity[i - batch_start] = (feature_vector == feature_vector[i]).astype(float)
            
        return batch_similarity
    
    def _compute_numerical_similarity(self, feature_vector, batch_start, batch_end):

        n_samples = len(feature_vector)
        batch_similarity = np.zeros((batch_end - batch_start, n_samples))
        

        feature_norm = (feature_vector - np.min(feature_vector)) / (np.max(feature_vector) - np.min(feature_vector))
        
        for i in range(batch_start, batch_end):
            batch_similarity[i - batch_start] = 1 - np.abs(feature_norm - feature_norm[i])
            
        return batch_similarity
    
    def compute_similarity_matrix(self, features, feature_types, feature_names=None):

        print("\nComputing Content-Based Similarity")
        print(f"Features shape: {features.shape}")
        if feature_names:
            print(f"Feature names: {feature_names}")
        
        n_samples = features.shape[0]
        n_batches = (n_samples - 1) // self.batch_size + 1
        
        similarity_matrix = np.zeros((n_samples, n_samples), dtype=np.float32)
        
        for feat_idx, (feat_type, feat_name) in enumerate(zip(feature_types, feature_names or [])):
            print(f"\nProcessing feature: {feat_name} ({feat_type})")
            feature_vector = features[:, feat_idx]
            
            weight = 1.0
            print(f"Feature weight: {weight}")
            
            start_time = time.time()
            print(f"Computing {feat_type} similarity for feature of length {len(feature_vector)}")
            
            for batch_idx in range(n_batches):
                batch_start = batch_idx * self.batch_size
                batch_end = min((batch_idx + 1) * self.batch_size, n_samples)
                
                if feat_type == 'categorical':
                    batch_similarity = self._compute_categorical_similarity(
                        feature_vector, batch_start, batch_end)
                else:
                    batch_similarity = self._compute_numerical_similarity(
                        feature_vector, batch_start, batch_end)
                
                similarity_matrix[batch_start:batch_end] += weight * batch_similarity
                
                progress = (batch_idx + 1) / n_batches * 100
                elapsed_time = time.time() - start_time
                print(f"Progress: {progress:.1f}%, Time elapsed: {elapsed_time:.2f}s")
        
        output_path = 'data/content_similarity.npy'
        np.save(output_path, similarity_matrix.astype(np.float32))
        print(f"\nSaved similarity matrix to {output_path}")
        
        return output_path