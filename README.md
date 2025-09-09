# Simple Fashion Recommendation System

A content-based fashion recommendation system that uses both product metadata and image features to recommend similar fashion items. The system combines traditional content-based filtering with K-means clustering to create a hybrid recommendation approach.

## Project Overview

This project implements a fashion recommendation system using the Fashion Product Images Dataset from Kaggle. The system analyzes fashion product attributes (category, color, gender, usage, etc.) along with visual features extracted from product images to recommend similar items to users.

### Key Features

- **Content-Based Filtering**: Recommends products based on item attributes like category, color, gender, and usage
- **Image Feature Extraction**: Uses color histogram analysis and PCA for visual similarity
- **K-means Clustering**: Groups similar products to improve recommendation diversity
- **Hybrid Similarity**: Combines content-based and clustering-based similarities with weighted scores
- **Comprehensive Evaluation**: Includes precision, recall, NDCG, diversity, and category accuracy metrics

## Project Structure

### Main Files

- **`main.py`**: Main entry point that orchestrates the complete recommendation pipeline from data loading to similarity matrix computation
- **`content_based_eval.ipynb`**: Jupyter notebook for evaluating the content-based recommendation system with various metrics
- **`notebook.ipynb`**: Interactive notebook for testing recommendations and visualizing results with product images

### Source Code (`src/` directory)

- **`data_processing.py`**: Handles loading and preprocessing of the fashion metadata CSV file
- **`feature_extraction.py`**: Extracts and normalizes categorical and numerical features from product metadata
- **`image_feature_extractor.py`**: Extracts color histogram features from product images and applies PCA for dimensionality reduction
- **`get_recommendations.py`**: Main recommendation engine that generates product recommendations using precomputed similarity matrices

### Model Components (`src/models/` directory)

- **`content_based.py`**: Implements content-based similarity computation using categorical and numerical feature matching
- **`kmeans_recommender.py`**: Custom K-means clustering implementation for grouping similar fashion products
- **`final_similarity_matrix.py`**: Combines content-based and clustering similarities with weighted averaging
- **`training_visualizer.py`**: Provides visualization tools for training metrics, cluster distributions, and processing statistics

## Requirements and Dependencies

### Core Dependencies
```
numpy>=1.19.0
pandas>=1.2.0
pillow>=8.0.0
matplotlib>=3.3.0
seaborn>=0.11.0
scikit-learn>=0.24.0 (optional, for comparison)
```

### Data Requirements
- Fashion Product Images Dataset (Small) from Kaggle
- CSV metadata file (`styles.csv`) with product information
- Product images directory (`data/images/`) containing JPG files named by product ID

## Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/HimasagarU/Simple-Fashion-Recommendation-System.git
cd Simple-Fashion-Recommendation-System
```

### 2. Create Virtual Environment
```bash
python -m venv fashion_rec_env
source fashion_rec_env/bin/activate  # On Windows: fashion_rec_env\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install numpy pandas pillow matplotlib seaborn
```

### 4. Prepare Data Structure
Create the following directory structure:
```
data/
├── styles.csv          # Product metadata file
└── images/             # Directory containing product images
    ├── 15970.jpg
    ├── 39386.jpg
    └── ...
```

### 5. Download Dataset
Download the Fashion Product Images Dataset (Small) from Kaggle and extract:
- Place `styles.csv` in the `data/` directory
- Extract all product images to `data/images/` directory

## Usage Instructions

### Running the Complete Pipeline

Execute the main pipeline to process data and compute similarity matrices:
```bash
python main.py
```

This will:
1. Load and process the metadata
2. Extract features from product attributes
3. Extract visual features from product images
4. Apply K-means clustering
5. Compute final hybrid similarity matrix
6. Save results to `data/final_similarity.npy`

### Getting Recommendations

Use the recommendation system in Python:
```python
from src.get_recommendations import RecommendationGenerator

# Initialize the recommendation system
recommender = RecommendationGenerator(
    similarity_matrix_path='data/final_similarity.npy',
    metadata_path='data/styles.csv'
)

# Get recommendations for a product
recommendations = recommender.get_recommendations(['39386'], k=10)
print(recommendations)
```

### Interactive Testing

Use the Jupyter notebook for interactive exploration:
```bash
jupyter notebook notebook.ipynb
```

This notebook allows you to:
- Test recommendations for specific products
- Visualize recommended items with images
- Explore similarity matrix statistics
- Analyze recommendation results

### Evaluation

Run the evaluation notebook to assess system performance:
```bash
jupyter notebook content_based_eval.ipynb
```

The evaluation includes:
- Precision and Recall at K
- Normalized Discounted Cumulative Gain (NDCG)
- Diversity scores
- Category matching accuracy
- Performance visualization

## System Parameters

### Feature Processing
- **Categorical features**: gender, masterCategory, subCategory, articleType, baseColour, season, usage
- **Numerical features**: year
- **Image features**: Color histograms with 32 bins per RGB channel, reduced to 50 dimensions using PCA

### Similarity Computation
- **Content-based weight**: 0.6
- **Clustering-based weight**: 0.4
- **K-means clusters**: 100
- **Batch processing**: 1000 samples per batch for memory efficiency

## Expected Performance

Based on evaluation results:
- **Precision@5**: ~0.12
- **Recall@5**: ~0.08
- **NDCG@5**: ~0.13
- **Category Accuracy**: ~0.96
- **Processing Time**: ~30-60 seconds for 44K products

## File Output

The system generates several files:
- `data/content_similarity.npy`: Content-based similarity matrix
- `data/final_similarity.npy`: Final hybrid similarity matrix
- `results/`: Directory containing evaluation plots and reports (if using visualizer)

## Troubleshooting

### Common Issues

1. **Memory Error**: Reduce batch_size parameters in model files
2. **Missing Images**: Ensure all referenced product IDs have corresponding JPG files
3. **Import Errors**: Verify all source files are in the correct directory structure
4. **Slow Performance**: Consider using smaller dataset subset for testing

### Performance Optimization

- Use smaller K-means cluster count for faster processing
- Reduce PCA components for image features
- Process data in smaller batches if memory is limited

## Contributing

To contribute to this project:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source. Please check the repository for specific license details.
