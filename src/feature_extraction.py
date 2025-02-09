import numpy as np
import pandas as pd

class FeatureExtractor:
    def __init__(self, columns=None):

        self.columns = columns or {
            'categorical': [],
            'numerical': []
        }
        self.category_mappings = {} 
        self.numerical_stats = {} 
        
    def _encode_categorical(self, series):
        unique_values = sorted(set(series.dropna()))
        mapping = {val: idx for idx, val in enumerate(unique_values)}
        mapping['unknown'] = len(mapping) 
        
        self.category_mappings[series.name] = mapping
        
        return series.map(lambda x: mapping.get(x, mapping['unknown']))
    
    def _normalize_numerical(self, series):
        series = series.astype(float)
        min_val = series.min()
        max_val = series.max()
        
        self.numerical_stats[series.name] = {
            'min': min_val,
            'max': max_val,
            'mean': series.mean()
        }
        
        if max_val == min_val:
            return np.zeros_like(series)
        
        return (series - min_val) / (max_val - min_val)
    
    def extract_features(self, metadata):
        print("\nExtracting features...")
        
        if isinstance(metadata, np.ndarray):
            metadata = pd.DataFrame(metadata)
        
        features = []
        
        for col in self.columns['categorical']:
            if col in metadata.columns:
                print(f"Processing categorical feature: {col}")

                metadata[col] = metadata[col].fillna('unknown')
                
                encoded = self._encode_categorical(metadata[col])
                features.append(encoded.values)
                
                n_categories = len(self.category_mappings[col])
                print(f"  - {n_categories} unique categories found")
        
        for col in self.columns['numerical']:
            if col in metadata.columns:
                print(f"Processing numerical feature: {col}")

                series = metadata[col]
                mean_val = series.mean()
                series = series.fillna(mean_val)
                
                normalized = self._normalize_numerical(series)
                features.append(normalized)

                stats = self.numerical_stats[col]
                print(f"  - Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
                print(f"  - Mean: {stats['mean']:.2f}")
        
        if not features:
            raise ValueError("No features were extracted. Check if column names match the metadata.")

        feature_matrix = np.column_stack(features)
        print(f"\nExtracted feature matrix shape: {feature_matrix.shape}")
        
        return feature_matrix
    
    def normalize_features(self, features):

        normalized = np.zeros_like(features, dtype=float)
        
        for i in range(features.shape[1]):
            col = features[:, i]
            min_val = np.min(col)
            max_val = np.max(col)
            
            if max_val == min_val:
                normalized[:, i] = 0
            else:
                normalized[:, i] = (col - min_val) / (max_val - min_val)
        
        return normalized
    
    def get_feature_info(self):

        info = {
            'categorical': {
                col: {
                    'n_categories': len(mappings),
                    'categories': list(mappings.keys())
                }
                for col, mappings in self.category_mappings.items()
            },
            'numerical': {
                col: stats
                for col, stats in self.numerical_stats.items()
            }
        }
        return info
