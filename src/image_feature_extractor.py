import numpy as np
from PIL import Image
import os
import time

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance_ratio = None
        
    def fit_transform(self, X):

        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        cov_matrix = np.cov(X_centered, rowvar=False)
        
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        self.components = eigenvectors[:, :self.n_components]
        total_var = np.sum(eigenvalues)
        self.explained_variance_ratio = eigenvalues[:self.n_components] / total_var
        

        return np.dot(X_centered, self.components)
    
    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

class ImageFeatureExtractor:
    def __init__(self, images_dir, n_bins=32, n_components=50):

        self.images_dir = images_dir
        self.n_bins = n_bins
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        
    def extract_color_histogram(self, image):
        image = image.convert('RGB')
        
        img_array = np.array(image)
        

        hist_features = []
        for channel in range(3):  
            hist, _ = np.histogram(
                img_array[:,:,channel], 
                bins=self.n_bins, 
                range=(0, 256), 
                density=True
            )
            hist_features.extend(hist)
        
        hist_features = np.array(hist_features)
        hist_features = hist_features / np.sum(hist_features)
        
        return hist_features
    
    def extract_features(self, product_ids):

        print("\nExtracting image features...")
        features = []
        processed_ids = []
        
        total_products = len(product_ids)
        start_time = time.time()
        last_update_time = start_time
        update_interval = 2  
        

        for idx, product_id in enumerate(product_ids):
            
            image_path = os.path.join(self.images_dir, f"{product_id}.jpg")
            
            if os.path.exists(image_path):
                try:

                    with Image.open(image_path) as img:

                        hist_features = self.extract_color_histogram(img)
                        features.append(hist_features)
                        processed_ids.append(product_id)
                        
                except Exception as e:
                    print(f"\nError processing image {product_id}: {str(e)}")
            else:
                print(f"\nImage not found for product {product_id}")
        
        print("\nImage processing completed!")
        
        if not features:
            raise ValueError("No valid images found")
            
        features = np.array(features)
        print(f"\nExtracted raw features shape: {features.shape}")
        
        print("Applying PCA dimension reduction...")
        reduced_features = self.pca.fit_transform(features)
        print(f"Reduced features shape: {reduced_features.shape}")

        cumulative_variance = np.cumsum(self.pca.explained_variance_ratio)
        print("\nExplained variance ratio:")
        print(f"First component: {self.pca.explained_variance_ratio[0]:.4f}")
        print(f"First 5 components: {np.sum(self.pca.explained_variance_ratio[:5]):.4f}")
        print(f"All {self.n_components} components: {np.sum(self.pca.explained_variance_ratio):.4f}")
        
        return reduced_features, processed_ids
    
    def save_features(self, features, output_path):
        np.save(output_path, features)
        print(f"\nSaved features to {output_path}")
        print(f"Feature matrix shape: {features.shape}") 