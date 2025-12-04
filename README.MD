🚀 Advanced Amazon Sales ETL-Analytics Pipeline

A professional-grade ETL pipeline for deep analysis of Amazon e-commerce sales data. This pipeline goes beyond simple cleaning, applying Machine Learning models and statistical analysis to generate actionable insights on customer behavior, sales trends, and product performance.

✨ Key Features
1️⃣ Customer Segmentation & Prediction

RFM Analysis: Computes Recency, Frequency, and Monetary value for each customer.

K-Means Clustering (K=4): Groups customers into value-based segments like Loyal, At-Risk, or New.

DBSCAN Clustering: Detects dense “core” loyal groups and identifies outliers.

High-Value Customer Prediction (XGBoost): Predicts top 20% customers by revenue and shows feature importance.

2️⃣ Anomaly & Risk Detection

Isolation Forest: Flags unusual transactions or outliers in RFM space.

Customer Churn Probability (RandomForest): Estimates churn risk for each customer.

3️⃣ Time Series & Comparative Analysis

Monthly Sales Trends: Tracks and forecasts orders and revenue trends.

Year-over-Year (YoY) Growth: Compares annual revenue and order counts to highlight growth patterns.

4️⃣ Product Insights

Custom Order Value Segmentation (OVS): Classifies customers into Low, Mid, or High segments based on purchase value quartiles.

Top Product Identification: Highlights the top 20 products generating the most revenue.

🛠 Prerequisites

Python 3.8+

Install dependencies:

pip install pandas numpy scikit-learn xgboost matplotlib seaborn colorama joblib

📂 Project Structure
Amazon Sales ETL Pipeline/
├── src/
│   └── amazon_etl_analytics.py      # Main ETL pipeline
├── Amazon.csv                        # Input sales data (mandatory)
└── outputs_amazon_advanced/          # Generated outputs
    ├── amazon_master_transformed.csv
    ├── customer_churn_probability.csv
    ├── monthly_sales_trend.csv
    ├── yearly_orders_summary.csv
    ├── top_products_summary.csv
    ├── *.pkl                         # Saved ML models (KMeans, DBSCAN, XGBoost, etc.)
    └── *.png                         # Generated visualizations

⚡ Usage

Place Data
Ensure Amazon.csv is in the project root (same level as src/).

Run the Pipeline

python src/amazon_etl_analytics.py


Review Outputs
All transformed data, model files, and charts are saved in outputs_amazon_advanced. Execution time and a summary will print in the terminal.

📊 Outputs Overview
Type	File Name	Description
Data	amazon_master_transformed.csv	Original data merged with RFM, scaled features, KMeans & DBSCAN clusters, OVS segment, anomaly flags, and high-value targets.
Data	customer_churn_probability.csv	Customer-level report including churn probability.
Data	monthly_sales_trend.csv	Monthly revenue and order counts with linear regression forecasts.
Data	yearly_orders_summary.csv	Annual revenue, order counts, and YoY growth percentages.
Data	top_products_summary.csv	Top 20 products ranked by total revenue.
Charts	L5_xgb_feature_importance.png	Feature importance chart for high-value customer prediction.
Charts	L6_monthly_revenue_trend.png	Monthly revenue trend vs forecast.
Charts	L7_yearly_orders_comparison.png	Annual orders comparison with YoY growth.
Charts	L8_customer_churn_probability.png	Distribution of churn probability across customers.
💡 Why Use This Pipeline

This project is designed for data-driven decision-making. It allows analysts and managers to:

Identify high-value customers and prioritize retention strategies.

Detect anomalies and potential fraud in sales transactions.

Track sales performance monthly and yearly.

Understand product performance and revenue contribution.

✅ Outcome: By the end of the pipeline, you’ll have cleaned data, actionable insights, ML predictions, and professional visualizations—all ready for reporting or further analysis.