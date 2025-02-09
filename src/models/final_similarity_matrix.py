import numpy as np
import time
import os

class FinalSimilarityComputer:
    def __init__(self, weights=(0.6, 0.4), batch_size=1000):

        self.weights = np.array(weights)
        self.weights = self.weights / np.sum(self.weights) 
        self.batch_size = batch_size
        
    def compute_final_similarity(self, content_similarity_path, clustering_similarity, cluster_labels):
        print("\nComputing Final Similarity Matrix...")
        start_time = time.time()
        

        content_similarity = np.load(content_similarity_path)
        

        final_similarity = (
            self.weights[0] * content_similarity +
            self.weights[1] * clustering_similarity
        )
        

        output_path = 'data/final_similarity.npy'
        np.save(output_path, final_similarity.astype(np.float32))
        
        print(f"Final similarity computation completed in {time.time() - start_time:.2f}s")
        return output_path
    
    def save_similarity_matrix(self, source_path, target_path):
        if source_path != target_path:
            import shutil
            shutil.move(source_path, target_path)
        print("Save completed")
    
    def load_similarity_matrix(self, filepath):
        return np.load(filepath)