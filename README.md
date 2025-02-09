# Simple-Fashion-Recommendation-System 

This repository contains the implementation of a **Fashion Recommendation System** using the [Fashion Product Images Dataset (Small) from Kaggle](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small). The project aims to recommend fashion items based on product attributes, user preferences.  

---

## 📊 Dataset  

The dataset contains 44,000+ images of fashion products, including categories like apparel, footwear, and accessories. Each product has the following metadata:
- **Product ID**
- **Category (e.g., Tops, Dresses, Footwear)**
- **Gender (Men, Women, Unisex)**
- **Color**
- **Usage (e.g., Casual, Formal, Sportswear)**
- **Price**
- **etc**
The combination of images and metadata enables a rich recommendation experience.

---

## ⚙️ Features  

- **Content-Based Filtering**: Recommends products based on item attributes like category, color, and price.  
- **Collaborative Filtering**: Suggests products based on user interactions and transaction history.  
- **Hybrid Model**: Merges both methods for more accurate recommendations.  
- **Evaluation Metrics**: Uses precision, recall, and F1-score to assess performance.

---

## 🚀 Installation  

1. Clone the repository:  
   ```bash
   git clone https://github.com/username/fashion-recommendation-system.git
   cd fashion-recommendation-system
   ```

2. Create a virtual environment and install dependencies:  
   ```bash
   python -m venv env
   source env/bin/activate  # For Linux/Mac
   env\Scripts\activate     # For Windows
   pip install -r requirements.txt
   ```

3. Run the project:  
   ```bash
   python src/main.py
   ```

---

## 🛠 Technologies  

- **Python**  
- **Pandas, NumPy**  
- **Matplotlib, Seaborn** for visualization  

---

If you want to customize further (e.g., add more details on the dataset preprocessing or recommendation model), let me know!
