# Furniture_Sales_Prediction

This project analyzes an E-commerce Furniture Dataset (2024) and builds predictive models to estimate the number of items sold (sold) based on product attributes such as:

Product Title (productTitle)

Original Price (originalPrice)

Discounted Price (price)

Discount Values (discount, discount_percent)

Tag/Shipping Info (tagText)

The goal is to uncover insights into sales patterns and build machine learning models to predict product demand.

# Dataset Description

The dataset contains 2,000 furniture products with the following columns:

Column	Description
productTitle	Name/description of the product
originalPrice	Original listed price
price	Final selling price
sold	Number of items sold
tagText	Shipping/offer information
discount	Absolute discount value
discount_percent	Discount percentage


# Exploratory Data Analysis (EDA)

Distribution of sales (sold) shows many products with low sales and a few with very high sales.

Most products are tagged as “Free shipping”, while others include paid shipping.

Discounts positively correlate with higher sales.

Price has a negative correlation with sales (lower-priced items sell more).

# Key Visualizations:

Sales distribution histogram

Scatter plots: Sales vs. Price, Sales vs. Discount %

Correlation heatmap (price, discount, sales)

Sales grouped by discount ranges

Sales grouped by shipping type

# Machine Learning Models

We trained multiple regression models to predict number of items sold:

1. Linear Regression

MAE: ~0.98

RMSE: ~1.22

R²: ~0.18

2. Random Forest Regressor

MAE: ~0.84

RMSE: ~1.06

R²: ~0.39 ✅ (Best performer)

3. XGBoost Regressor

MAE: ~0.83

RMSE: ~1.08

R²: ~0.36

# Best Model: Random Forest Regressor (highest R², lowest error).

 # Results & Insights

Discounts and lower prices drive higher sales.

Free shipping products dominate sales.

Machine learning models achieved moderate accuracy (R² ~0.39).

Random Forest performed best, capturing non-linear relationships.

# Future Improvements

Perform hyperparameter tuning (GridSearchCV/RandomizedSearchCV) for Random Forest & XGBoost.

Use text features from productTitle (via NLP / TF-IDF embeddings).

Add more categorical encoding for tagText and shipping costs.

Try neural networks for improved performance.

🛠# Tech Stack

Python 3.12

Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost

Jupyter Notebook for implementation

# How to Run

Clone this repo / download the notebook.

Install dependencies:

pip install pandas numpy matplotlib seaborn scikit-learn xgboost


Open the notebook:

jupyter notebook "E-commerce Furniture Dataset 2024.ipynb"


Run all cells to reproduce results.
