import numpy as np
import pandas as pd
import os
import time
from src.data_processing import DataProcessor
from src.feature_extraction import FeatureExtractor
from src.image_feature_extractor import ImageFeatureExtractor
from src.models.kmeans_recommender import KMeansRecommender
from src.models.final_similarity_matrix import FinalSimilarityComputer

def main():
    try:
        start_time = time.time()
        
        print("Loading and processing data...")
        data_processor = DataProcessor('data/styles.csv')
        metadata, headers = data_processor.load_metadata()
        
        if metadata is None or len(metadata) == 0:
            raise ValueError("Failed to load metadata")
            
        print(f"Loaded metadata shape: {metadata.shape}")
        print(f"Headers: {headers}")
        
        # feature columns
        feature_columns = {
            'categorical': ['gender', 'masterCategory', 'subCategory', 
                           'articleType', 'baseColour', 'season', 'usage'],
            'numerical': ['year']
        }
        
        missing_columns = [col for col in feature_columns['categorical'] + feature_columns['numerical'] 
                         if col not in metadata.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Extract metadata features
        print("\nExtracting features...")
        feature_extractor = FeatureExtractor(columns=feature_columns)
        metadata_features = feature_extractor.extract_features(metadata)
        
        if metadata_features is None or len(metadata_features) == 0:
            raise ValueError("Failed to extract features")
            
        print(f"Extracted feature matrix shape: {metadata_features.shape}")
        
        # Extract image features
        print("\nExtracting image features...")
        image_extractor = ImageFeatureExtractor(
            images_dir='data/images',
            n_bins=32,
            n_components=50
        )
        image_features = image_extractor.extract_features(metadata['id'].tolist())
        
        if image_features is not None:
            try:

                image_features = np.array(image_features[0])
                
                if len(image_features.shape) > 2:
                    n_samples = image_features.shape[0]
                    image_features = image_features.reshape(n_samples, -1)
                
                print(f"Image features shape after reshaping: {image_features.shape}")
                
                full_image_features = np.zeros((len(metadata), image_features.shape[1]), dtype=np.float32)
                
                valid_indices = metadata['id'].isin(metadata['id'].iloc[:len(image_features)]).values
                
                full_image_features[valid_indices] = image_features
                
                print(f"Full image features shape: {full_image_features.shape}")
                
                # Normalize features
                metadata_features = np.array(metadata_features, dtype=np.float32)
                
                metadata_features_norm = metadata_features / (np.linalg.norm(metadata_features, axis=1)[:, np.newaxis] + 1e-8)
                image_features_norm = full_image_features / (np.linalg.norm(full_image_features, axis=1)[:, np.newaxis] + 1e-8)
                
                image_features_norm = np.nan_to_num(image_features_norm)
                
                # Combine features
                combined_features = np.hstack([
                    metadata_features_norm * 0.7,
                    image_features_norm * 0.3
                ])
                
                print(f"Combined features shape: {combined_features.shape}")
                
            except Exception as e:
                print(f"Error processing image features: {e}")
                print("Falling back to metadata features only...")
                combined_features = metadata_features / (np.linalg.norm(metadata_features, axis=1)[:, np.newaxis] + 1e-8)
        else:
            print("\nUsing only metadata features...")
            combined_features = metadata_features / (np.linalg.norm(metadata_features, axis=1)[:, np.newaxis] + 1e-8)
        
        print(f"Final feature matrix shape: {combined_features.shape}")
        
        print("\nComputing clustering-based similarity...")
        clustering_start = time.time()
        kmeans_rec = KMeansRecommender(
            n_clusters=100,
            max_iter=100,      
            batch_size=1000    
        )
        kmeans_rec.fit(combined_features)
        
        clustering_similarity = kmeans_rec.compute_similarity_matrix()
        print(f"Clustering similarity computation time: {time.time() - clustering_start:.2f}s")
        
        # Compute final similarity
        print("\nComputing final similarity...")
        final_start = time.time()
        final_computer = FinalSimilarityComputer(
            weights=(0.6, 0.4),  
            batch_size=1000
        )
        
        content_similarity_path = 'data/content_similarity.npy'
        
        final_similarity_path = final_computer.compute_final_similarity(
            content_similarity_path,
            clustering_similarity,
            kmeans_rec.labels
        )
        print(f"Final similarity computation time: {time.time() - final_start:.2f}s")
        
        print("\nSaving final similarity matrix...")
        final_computer.save_similarity_matrix(
            final_similarity_path,
            'data/final_similarity.npy'
        )
        
        total_time = time.time() - start_time
        print(f"\nPipeline completed successfully in {total_time:.2f}s!")
        
    except FileNotFoundError as e:
        print(f"File not found error: {e}")
    except ValueError as e:
        print(f"Value error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 