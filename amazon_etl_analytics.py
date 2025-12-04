# =============================================================================
# AMAZON SALES ANALYTICS ETL PIPELINE - LEVEL NA+
# File: src/amazon_etl_analytics.py
# Purpose: Unique, advanced ETL for Amazon sales data including RFM, Custom OVS,
#          Isolation Forest Anomaly Detection, DBSCAN/KMeans Clustering,
#          Time Series Trends, Churn Prediction, and High-Value Customer Prediction.
# Requires: Amazon.csv (uploaded)
# Produces: Transformed CSV, Models (XGBoost, RF Churn), and High-Quality Charts.
# =============================================================================

# -------------------------
# 0. IMPORTS & SETUP
# -------------------------
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from colorama import init as colorama_init
from colorama import Fore, Style
import joblib
import warnings

# Suppress minor Seaborn FutureWarning messages for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Colorama Setup (Professional Aesthetic) ---
colorama_init(autoreset=True)
STAGE = Fore.CYAN + Style.BRIGHT
PROCESS = Fore.GREEN
SUCCESS = Fore.MAGENTA + Style.BRIGHT
WARN = Fore.YELLOW
ERROR = Fore.RED + Style.BRIGHT
RESET = Style.RESET_ALL

# -------------------------
# 1. CONFIGURATION & PATHS
# -------------------------
# Assuming Amazon.csv is available in the project root directory
DATA_FILE = 'Amazon.csv'
# BASE_DIR is '.../src'. We calculate the PROJECT_ROOT by going up one level.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, '..')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs_amazon_advanced')
os.makedirs(OUTPUTS_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", palette="viridis")  # Unique seaborn style

# Global references for output files
MASTER_CSV = os.path.join(OUTPUTS_DIR, 'amazon_master_transformed.csv')
KMEANS_PKL = os.path.join(OUTPUTS_DIR, 'kmeans_model.pkl')
DBSCAN_PKL = os.path.join(OUTPUTS_DIR, 'dbscan_model.pkl')
XGB_PKL = os.path.join(OUTPUTS_DIR, 'xgb_high_value_predictor.pkl')
SCALER_PKL = os.path.join(OUTPUTS_DIR, 'feature_scaler.pkl')


# -------------------------
# 2. ETL STAGE FUNCTIONS
# -------------------------

def extract_data(file_path):
    """E: Reads the raw CSV data."""
    print(STAGE + "\n[E] ━━━━━━━━━━ EXTRACT STAGE ━━━━━━━━━━" + RESET)
    try:
        # Correct local file loading
        csv_path = os.path.join(BASE_DIR, file_path)
        df = pd.read_csv(csv_path)

        print(PROCESS + f"Successfully loaded {len(df)} records." + RESET)
        return df
    except FileNotFoundError:
        print(ERROR + f"Data file '{csv_path}' not found." + RESET)
        raise


def transform_data(df):
    """
    T: Performs data cleaning, RFM calculation, feature engineering, and modeling.
    Returns: df_final, metrics, ml_features, monthly_sales_trend, yearly_orders_df, rfm_churn_df, top_products_df
    """
    print(STAGE + "\n[T] ━━━━━━━━━━ TRANSFORM STAGE ━━━━━━━━━━" + RESET)

    # Initialize metrics dictionary that will be returned
    metrics = {'report': {}}

    # --- T1. Cleaning & Initial Feature Engineering ---
    print(PROCESS + "T1: Initial Cleaning and Feature Engineering..." + RESET)
    df.columns = [col.replace(' ', '') for col in df.columns]

    # Drop rows with missing CustomerID or essential sales metrics
    df.dropna(subset=['CustomerID', 'TotalAmount', 'OrderDate', 'Quantity'], inplace=True)

    # Convert types and filter invalid/cancelled transactions
    df['OrderDate'] = pd.to_datetime(df['OrderDate'])
    df['TotalAmount'] = pd.to_numeric(df['TotalAmount'], errors='coerce')
    df = df[df['TotalAmount'] > 0]
    df = df[df['OrderStatus'] != 'Cancelled']

    # --- T2. RFM Calculation (Recency, Frequency, Monetary) ---
    print(PROCESS + "T2: Computing RFM metrics..." + RESET)

    # Use the max date in the data as the reference point
    reference_date = df['OrderDate'].max() + pd.Timedelta(days=1)

    rfm_df = df.groupby('CustomerID').agg(
        Recency=('OrderDate', lambda x: (reference_date - x.max()).days),
        Frequency=('OrderID', 'nunique'),
        Monetary=('TotalAmount', 'sum')
    ).reset_index()

    # --- T3. Anomaly Detection (Isolation Forest) ---
    # Identify and flag outliers in the RFM space before scaling
    print(PROCESS + "T3: Running Isolation Forest for Anomaly Detection (Outlier Flagging)..." + RESET)
    iso_cols = ['Recency', 'Frequency', 'Monetary']
    iso_model = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
    rfm_df['Anomaly'] = iso_model.fit_predict(rfm_df[iso_cols])
    rfm_df['IsOutlier'] = rfm_df['Anomaly'].apply(lambda x: 1 if x == -1 else 0)
    print(WARN + f"Found {rfm_df['IsOutlier'].sum()} outliers (5% contamination rate)." + RESET)

    # --- T4. Feature Scaling (Robust Scaling) ---
    print(PROCESS + "T4: Scaling RFM Features using StandardScaler..." + RESET)
    rfm_scaled = rfm_df.copy()
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(rfm_scaled[iso_cols])

    rfm_scaled['R_Scaled'] = scaled_features[:, 0]
    rfm_scaled['F_Scaled'] = scaled_features[:, 1]
    rfm_scaled['M_Scaled'] = scaled_features[:, 2]
    joblib.dump(scaler, SCALER_PKL)

    # --- T5. Advanced Clustering: KMeans for Main Segments ---
    print(PROCESS + "T5: Applying K-Means Clustering for primary segmentation..." + RESET)
    K = 4
    kmeans = KMeans(n_clusters=K, n_init=10, random_state=42)
    rfm_scaled['KMeans_Cluster'] = kmeans.fit_predict(rfm_scaled[['R_Scaled', 'F_Scaled', 'M_Scaled']])
    joblib.dump(kmeans, KMEANS_PKL)
    print(SUCCESS + f"KMeans (K={K}) model trained and saved." + RESET)

    # --- T6. Advanced Clustering: DBSCAN for Density-Based Clusters ---
    print(PROCESS + "T6: Applying DBSCAN for density-based grouping (identifying core loyal groups)..." + RESET)
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    rfm_scaled['DBSCAN_Cluster'] = dbscan.fit_predict(rfm_scaled[['R_Scaled', 'F_Scaled', 'M_Scaled']])
    joblib.dump(dbscan, DBSCAN_PKL)
    print(
        SUCCESS + f"DBSCAN model trained and saved. Found {rfm_scaled['DBSCAN_Cluster'].nunique()} density clusters (including noise -1)." + RESET)

    # --- T7. Custom Feature: Order Value Segmentation (OVS) ---
    print(PROCESS + "T7: Applying Custom Order Value Segmentation (OVS)..." + RESET)
    mon_q = rfm_scaled['Monetary'].quantile([0.25, 0.75])

    def ovs_segment(monetary):
        if monetary >= mon_q.iloc[1]:
            return 'OVS_High'
        elif monetary <= mon_q.iloc[0]:
            return 'OVS_Low'
        else:
            return 'OVS_Mid'

    rfm_scaled['OVS_Segment'] = rfm_scaled['Monetary'].apply(ovs_segment)

    # --- T8. Target Creation for ML: High-Value Customer Prediction ---
    print(PROCESS + "T8: Creating binary target for XGBoost (Top 20% Monetary)..." + RESET)
    monetary_threshold = rfm_scaled['Monetary'].quantile(0.80)
    rfm_scaled['IsHighValue'] = (rfm_scaled['Monetary'] >= monetary_threshold).astype(int)

    # --- T9. Predictive Modeling: XGBoost Classification ---
    print(PROCESS + "T9: Training XGBoost Classifier to predict High-Value Customers..." + RESET)

    # Define base features
    ml_features = ['R_Scaled', 'F_Scaled', 'M_Scaled', 'KMeans_Cluster']

    # Add dummy variables for OVS segments (excluding one for linearity)
    rfm_ml = pd.get_dummies(rfm_scaled, columns=['OVS_Segment'], drop_first=True)
    # Capture the full list of features used in the model, including dummies
    ml_features += [col for col in rfm_ml.columns if col.startswith('OVS_Segment_')]

    X = rfm_ml[ml_features]
    y = rfm_ml['IsHighValue']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    xgb_model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )

    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)

    # Generate classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = report['accuracy']

    joblib.dump(xgb_model, XGB_PKL)

    # Populate the metrics dictionary with model results
    metrics['report']['accuracy'] = accuracy
    metrics['report']['report_dict'] = report

    print(SUCCESS + f"XGBoost Model trained. Test Accuracy: {accuracy:.4f}." + RESET)

    # --- T10. Time Series Trend Analysis (Linear Regression) ---
    print(PROCESS + "T10: Performing Time Series Trend Analysis (Monthly Orders and Revenue)..." + RESET)
    monthly_orders = df.groupby(df['OrderDate'].dt.to_period('M')).agg(
        OrderCount=('OrderID', 'nunique'),
        TotalRevenue=('TotalAmount', 'sum')
    ).reset_index()

    monthly_orders['OrderDate'] = monthly_orders['OrderDate'].dt.to_timestamp()
    monthly_orders['MonthNum'] = np.arange(len(monthly_orders))

    # Predict order trend using Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(monthly_orders[['MonthNum']], monthly_orders['OrderCount'])
    monthly_orders['PredictedOrders'] = lr_model.predict(monthly_orders[['MonthNum']])
    joblib.dump(lr_model, os.path.join(OUTPUTS_DIR, 'lr_order_trend_model.pkl'))
    print(SUCCESS + "Monthly order trends calculated and linear model trained." + RESET)

    # --- T11. Customer Churn Probability Prediction (RandomForestClassifier) ---
    print(PROCESS + "T11: Predicting Customer Churn Probability (RandomForestClassifier)..." + RESET)
    # Target: 1 if customer is low frequency (only 1 order), 0 otherwise
    rfm_ml['IsLowFrequency'] = (rfm_ml['Frequency'] < 2).astype(int)
    churn_features = ['Recency', 'Frequency', 'Monetary', 'KMeans_Cluster']

    X_churn = rfm_ml[churn_features]
    y_churn = rfm_ml['IsLowFrequency']

    rfm_churn_df = rfm_scaled[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'KMeans_Cluster']].copy()

    rf_churn_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_churn_clf.fit(X_churn, y_churn)

    rfm_churn_df['Churn_Prob'] = rf_churn_clf.predict_proba(X_churn)[:, 1]
    joblib.dump(rf_churn_clf, os.path.join(OUTPUTS_DIR, 'rf_churn_predictor.pkl'))
    print(SUCCESS + "RandomForest Churn Probability model trained and probabilities calculated." + RESET)

    # --- T12. Year-over-Year (YoY) Comparison ---
    print(PROCESS + "T12: Calculating Year-over-Year sales and order comparison..." + RESET)
    df['Year'] = df['OrderDate'].dt.year
    yearly_orders_df = df.groupby('Year').agg(
        TotalRevenue=('TotalAmount', 'sum'),
        TotalOrders=('OrderID', 'nunique')
    ).reset_index()
    yearly_orders_df['YoY_Revenue_Growth_%'] = yearly_orders_df['TotalRevenue'].pct_change().mul(100).round(2)
    yearly_orders_df['YoY_Orders_Growth_%'] = yearly_orders_df['TotalOrders'].pct_change().mul(100).round(2)
    print(SUCCESS + "YoY growth metrics calculated." + RESET)

    # --- T13. Product-Level Analytics ---
    print(PROCESS + "T13: Identifying Top 20 Products by Revenue..." + RESET)
    top_products_df = df.groupby('ProductName').agg(
        TotalRevenue=('TotalAmount', 'sum'),
        TotalQuantity=('Quantity', 'sum'),
        UniqueOrders=('OrderID', 'nunique')
    ).sort_values(by='TotalRevenue', ascending=False).head(20).reset_index()
    print(SUCCESS + "Top 20 products identified." + RESET)

    # Re-merge the original data with the RFM/ML features
    final_df = df.merge(rfm_scaled[['CustomerID', 'Recency', 'Frequency', 'Monetary',
                                    'KMeans_Cluster', 'DBSCAN_Cluster', 'OVS_Segment',
                                    'IsOutlier', 'IsHighValue']],
                        on='CustomerID', how='left')

    print(SUCCESS + "Transformation stage complete. All features and models processed." + RESET)

    # Updated return to include all summary dataframes
    return (
        final_df, metrics, ml_features, monthly_orders, yearly_orders_df, rfm_churn_df, top_products_df
    )


def load_artifacts(df_final, metrics, monthly_sales_trend, yearly_orders_df, rfm_churn_df, top_products_df):
    """L: Saves the final data, summary tables, and generates visualizations."""
    print(STAGE + "\n[L] ━━━━━━━━━━ LOAD STAGE ━━━━━━━━━━" + RESET)

    # --- L1. Save Master Transformed Data ---
    df_final.to_csv(MASTER_CSV, index=False)
    print(PROCESS + f"L1: Master transformed data saved to: {MASTER_CSV}" + RESET)

    # --- L2. Save New Summary CSVs ---
    print(PROCESS + "L2: Saving new summary data tables..." + RESET)
    monthly_sales_trend.to_csv(os.path.join(OUTPUTS_DIR, 'monthly_sales_trend.csv'), index=False)
    yearly_orders_df.to_csv(os.path.join(OUTPUTS_DIR, 'yearly_orders_summary.csv'), index=False)
    # Only export necessary churn columns for a clean report
    rfm_churn_df[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'Churn_Prob', 'KMeans_Cluster']].to_csv(
        os.path.join(OUTPUTS_DIR, 'customer_churn_probability.csv'), index=False)
    top_products_df.to_csv(os.path.join(OUTPUTS_DIR, 'top_products_summary.csv'), index=False)
    print(SUCCESS + "New summary CSVs saved." + RESET)

    # --- L3. Generate and Save Charts (8 total plots now) ---
    print(PROCESS + "L3: Generating advanced analytical plots..." + RESET)

    # Use unique customer rows for customer-level analytics
    rfm_plot_df = df_final.drop_duplicates(subset=['CustomerID']).copy()

    # Merge Churn Prob back to the plotting DF
    rfm_plot_df = rfm_plot_df.merge(rfm_churn_df[['CustomerID', 'Churn_Prob']], on='CustomerID', how='left')

    # --- Plot A: 3D RFM Plot (KMeans Clusters) --- (Original L3)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(rfm_plot_df['Recency'], rfm_plot_df['Frequency'],
                         np.log1p(rfm_plot_df['Monetary']),
                         c=rfm_plot_df['KMeans_Cluster'], cmap='magma', s=50)
    ax.set_title('3D RFM Segmentation (KMeans)', fontsize=16)
    ax.set_xlabel('Recency (Days)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_zlabel('Monetary (Log Transformed)', fontsize=12)
    fig.colorbar(scatter, label='KMeans Cluster ID')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, 'L3_rfm_kmeans_3d.png'))
    plt.close()
    print(SUCCESS + "3D RFM KMeans plot saved." + RESET)

    # --- Plot B: DBSCAN Cluster Distribution vs. Anomaly --- (Original L4)
    plt.figure(figsize=(10, 6))
    sns.countplot(data=rfm_plot_df, x='DBSCAN_Cluster', hue='IsOutlier', palette='Pastel1')
    plt.title('DBSCAN Cluster Size & Anomaly Breakdown', fontsize=16)
    plt.xlabel('DBSCAN Cluster ID (-1 is Noise/Outlier)', fontsize=12)
    plt.ylabel('Customer Count', fontsize=12)
    plt.legend(title='Is Outlier')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, 'L4_dbscan_anomaly_count.png'))
    plt.close()
    print(SUCCESS + "DBSCAN/Anomaly distribution plot saved." + RESET)

    # --- Plot C: XGBoost Feature Importance --- (Original L5)
    # The keys 'feature_importances' and 'feature_names' are added in run_amazon_pipeline
    if 'feature_importances' in metrics['report'] and 'feature_names' in metrics['report']:
        plt.figure(figsize=(10, 6))
        feature_importance = pd.Series(metrics['report']['feature_importances'],
                                       index=metrics['report']['feature_names']).sort_values(ascending=False)
        sns.barplot(x=feature_importance, y=feature_importance.index, palette='coolwarm',
                    hue=feature_importance.index, legend=False)
        plt.title('XGBoost Feature Importance for High-Value Prediction ', fontsize=16)
        plt.xlabel('F-Score (Importance)', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUTS_DIR, 'L5_xgb_feature_importance.png'))
        plt.close()
        print(SUCCESS + "XGBoost Feature Importance chart saved." + RESET)
    else:
        print(WARN + "Skipping XGBoost Feature Importance plot: Feature data not available." + RESET)

    # --- Plot D: Monthly Revenue Trend (NEW) ---
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_sales_trend['OrderDate'], monthly_sales_trend['TotalRevenue'], marker='o', linestyle='-',
             label='Actual Monthly Revenue', color='darkblue')
    # Plot predicted order trend line (scaled by average order value for comparison)
    avg_order_value = monthly_sales_trend['TotalRevenue'].sum() / monthly_sales_trend['OrderCount'].sum()
    plt.plot(monthly_sales_trend['OrderDate'],
             monthly_sales_trend['PredictedOrders'] * avg_order_value,
             linestyle='--', color='red', label='Linear Order Trend (Scaled to Revenue)')
    plt.title(
        'Monthly Revenue Trend & Linear Order Forecast [Image of a time series plot showing actual and forecasted revenue]',
        fontsize=16)
    plt.xlabel('Order Date', fontsize=12)
    plt.ylabel('Total Revenue', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, 'L6_monthly_revenue_trend.png'))
    plt.close()
    print(SUCCESS + "Monthly Revenue Trend chart saved." + RESET)

    # --- Plot E: Year-over-Year Orders Comparison (NEW) ---
    plt.figure(figsize=(10, 6))
    sns.barplot(data=yearly_orders_df, x='Year', y='TotalOrders', palette='cool', hue='Year', legend=False)
    for index, row in yearly_orders_df.iterrows():
        if index > 0:
            plt.text(row.name, row.TotalOrders,
                     f"{row['YoY_Orders_Growth_%']:.1f}%",
                     color='green' if row['YoY_Orders_Growth_%'] > 0 else 'red',
                     ha="center", va="bottom", weight='bold')
    plt.title('Yearly Total Orders Comparison & Growth Rate ', fontsize=16)
    plt.ylabel('Total Orders Count', fontsize=12)
    plt.xlabel('Year', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, 'L7_yearly_orders_comparison.png'))
    plt.close()
    print(SUCCESS + "Yearly Orders Comparison chart saved." + RESET)

    # --- Plot F: Predicted Customer Churn Probability Distribution (NEW) ---
    plt.figure(figsize=(10, 6))
    sns.histplot(rfm_plot_df['Churn_Prob'], bins=25, kde=True, color='lightcoral')
    plt.title('Predicted Customer Churn Probability Distribution ', fontsize=16)
    plt.xlabel('Churn Probability (0 = Low Churn Risk, 1 = High Churn Risk)', fontsize=12)
    plt.ylabel('Number of Customers', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, 'L8_customer_churn_probability.png'))
    plt.close()
    print(SUCCESS + "Customer Churn Probability Distribution saved." + RESET)

    # --- Plot G: Top Products by Revenue (NEW) ---
    plt.figure(figsize=(10, 8))
    sns.barplot(x='TotalRevenue', y='ProductName', data=top_products_df, palette='Spectral', hue='ProductName',
                legend=False)
    plt.title('Top 20 Products by Total Revenue', fontsize=16)
    plt.xlabel('Total Revenue', fontsize=12)
    plt.ylabel('Product Name', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, 'L9_top_products_revenue.png'))
    plt.close()
    print(SUCCESS + "Top Products by Revenue chart saved." + RESET)

    # --- Plot H: OVS Segment Monetary Box Plot (Original L6) ---
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=rfm_plot_df, x='OVS_Segment', y='Monetary', showfliers=False,
                order=['OVS_Low', 'OVS_Mid', 'OVS_High'])
    plt.title('Monetary Distribution by Custom OVS Segment (Flier-Free) ', fontsize=16)
    plt.xlabel('Order Value Segment', fontsize=12)
    plt.ylabel('Monetary Value', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, 'L10_ovs_monetary_boxplot.png'))
    plt.close()
    print(SUCCESS + "OVS Monetary boxplot saved." + RESET)

    print(SUCCESS + "\n[L] Load Stage Finalized. All artifacts are ready." + RESET)


# -------------------------
# 5. ORCHESTRATION
# -------------------------

def run_amazon_pipeline():
    """Executes the full ETL-Analytics workflow."""
    start_time = time.time()

    print(SUCCESS + "\n\n🚀 Starting Unique Amazon Sales ETL-Analytics Pipeline 🚀" + RESET)

    try:
        # E: Extract
        df_raw = extract_data(DATA_FILE)

        # T: Transform (Unpack 7 return values including new trend/churn DFs)
        # We rename the 3rd returned value from 'ml_features' to 'feature_columns' locally
        (df_final, metrics, feature_columns,
         monthly_sales_trend, yearly_orders_df,
         rfm_churn_df, top_products_df) = transform_data(df_raw)

        # Load trained model to extract feature importances
        xgb_model = joblib.load(XGB_PKL)

        # Auto-inject the correct feature names & importances into metrics for plotting
        metrics['report']['feature_names'] = feature_columns
        metrics['report']['feature_importances'] = list(xgb_model.feature_importances_)

        # L: Load (saves data, models, plots) - Pass all new summary DFs
        load_artifacts(df_final, metrics, monthly_sales_trend, yearly_orders_df, rfm_churn_df, top_products_df)

    except Exception as e:
        print(ERROR + f"\n🔴 CRITICAL PIPELINE FAILURE: {e}" + RESET)
        print(ERROR + "Check the error details above, particularly file path errors for Amazon.csv." + RESET)

    end_time = time.time()
    duration = end_time - start_time

    print(SUCCESS + "\n✅ ETL Pipeline Completed Successfully!" + RESET)
    print(SUCCESS + f"Total Execution Time: {duration:.2f} seconds." + RESET)
    print(SUCCESS + f"Outputs directory: {OUTPUTS_DIR}" + RESET)
    print(SUCCESS + "--------------------------------------------------------" + RESET)


if __name__ == "__main__":
    run_amazon_pipeline()