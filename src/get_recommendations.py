import numpy as np
import pandas as pd

class RecommendationGenerator:
    def __init__(self, similarity_matrix_path, metadata_path):

        print("Loading similarity matrix and metadata...")
        self.similarity_matrix = np.load(similarity_matrix_path)
        self.metadata = pd.read_csv(metadata_path)

        print("Creating product ID mapping...")
        self.product_indices = {
            str(pid): idx for idx, pid in enumerate(self.metadata['id'].astype(str))
        }
        
        print(f"Loaded similarity matrix shape: {self.similarity_matrix.shape}")
        print(f"Number of products in metadata: {len(self.metadata)}")
        print(f"Number of mapped product IDs: {len(self.product_indices)}")
        
    def get_recommendations(self, product_ids, k=5, filters=None):

        recommendations = {}
        
        for product_id in product_ids:
            product_id = str(product_id) 
            
            if product_id not in self.product_indices:
                print(f"Warning: Product ID {product_id} not found in database")
                continue
            
            product_idx = self.product_indices[product_id]
            
            if product_idx >= len(self.similarity_matrix):
                print(f"Warning: Product index {product_idx} out of bounds for similarity matrix")
                continue
                
            similarities = self.similarity_matrix[product_idx]
            
            similar_indices = np.argsort(similarities)[::-1]
            
            similar_indices = similar_indices[similar_indices != product_idx]
            
            if filters:
                mask = np.ones(len(similar_indices), dtype=bool)
                for key, value in filters.items():
                    if key in self.metadata.columns:
                        mask &= self.metadata.iloc[similar_indices][key] == value
                similar_indices = similar_indices[mask]
            
            top_k_indices = similar_indices[:k]
            
            recommendations[product_id] = {
                'recommendations': [
                    {
                        'product_id': str(self.metadata.iloc[idx]['id']),
                        'similarity_score': similarities[idx],
                        'metadata': self.metadata.iloc[idx].to_dict()
                    }
                    for idx in top_k_indices
                ]
            }
        
        return recommendations 