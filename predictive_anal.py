import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import os
from prep_clean import load_and_clean_data
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings

# Define output directory
OUTPUT_DIR = 'PRED_ANAL_OUTPUT'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def write_to_report(file, text, section=None):
    """Helper function to write to report file"""
    if section:
        file.write(f"\n{section}\n")
        file.write("=" * len(section) + "\n")
    file.write(f"{text}\n")

def forecast_compost_output(df, report_file):
    """
    Time series forecasting for compost production using ARIMA
    Uses all historical data for training but focuses visualization on 2024 onwards
    """
    try:
        # Prepare time series data
        df['date'] = pd.to_datetime(df['date_collection_datetime'])
        
        # Create full time series for training
        ts_data_full = df.groupby('date')['compost_created_lbs'].sum().asfreq('D').fillna(method='ffill')
        
        # Fit ARIMA model on full dataset
        print("Fitting ARIMA model...")
        model = ARIMA(ts_data_full, order=(1,1,1))
        results = model.fit()
        
        # Make predictions for next 120 days
        forecast = results.forecast(steps=120)
        
        # Filter recent data for visualization and metrics
        ts_data_recent = ts_data_full['2024-01-01':]
        
        # Calculate forecast metrics using last 120 days of actual data
        actual_last_120 = ts_data_full[-120:]
        forecast_first_120 = forecast[:120]
        mae = mean_absolute_error(actual_last_120, forecast_first_120)
        mse = mean_squared_error(actual_last_120, forecast_first_120)
        rmse = np.sqrt(mse)
        
        # Calculate additional metrics for comprehensive evaluation
        mape = np.mean(np.abs((actual_last_120 - forecast_first_120) / actual_last_120)) * 100
        r2 = r2_score(actual_last_120, forecast_first_120)
        
        # Write detailed metrics to report
        write_to_report(report_file, f"\nModel Performance Metrics:")
        write_to_report(report_file, f"Mean Absolute Error (MAE): {mae:.2f} lbs")
        write_to_report(report_file, f"Mean Squared Error (MSE): {mse:.2f} lbs²")
        write_to_report(report_file, f"Root Mean Squared Error (RMSE): {rmse:.2f} lbs")
        write_to_report(report_file, f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        write_to_report(report_file, f"R-squared Score: {r2:.4f}")
        
        # Write results to report
        write_to_report(report_file, "Time Series Forecast Analysis", "Forecast Results")
        write_to_report(report_file, f"Training Data Period: {ts_data_full.index.min().strftime('%Y-%m-%d')} to {ts_data_full.index.max().strftime('%Y-%m-%d')}")
        write_to_report(report_file, f"Forecast Period: Next 120 days")
        write_to_report(report_file, f"\nForecast Statistics:")
        write_to_report(report_file, f"Average Daily Compost: {forecast.mean():.2f} lbs")
        write_to_report(report_file, f"Forecast Range: {forecast.min():.2f} - {forecast.max():.2f} lbs")
        write_to_report(report_file, f"Standard Deviation: {forecast.std():.2f} lbs")
        write_to_report(report_file, f"\nModel Performance:")
        write_to_report(report_file, f"Mean Absolute Error: {mae:.2f}")
        write_to_report(report_file, f"Mean Squared Error: {mse:.2f}")
        write_to_report(report_file, f"Root Mean Squared Error: {np.sqrt(mse):.2f}")
        
        # Plot results focusing on recent data
        plt.figure(figsize=(15, 8))
        
        # Plot recent actual data (2024 onwards)
        ts_data_recent.plot(label='Historical Data (2024+)', color='blue')
        
        # Plot forecast with confidence intervals
        forecast.plot(label='120-day Forecast', color='red', linestyle='--')
        
        # Add confidence intervals
        conf_int = results.get_forecast(steps=120).conf_int()
        plt.fill_between(conf_int.index, 
                        conf_int.iloc[:, 0], 
                        conf_int.iloc[:, 1], 
                        color='red', 
                        alpha=0.1, 
                        label='95% Confidence Interval')
        
        plt.title('Compost Production Forecast\n(Showing 2024 onwards, model trained on full history)')
        plt.xlabel('Date')
        plt.ylabel('Compost Created (lbs)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'compost_forecast.png'))
        plt.close()
        
        return forecast
    except Exception as e:
        write_to_report(report_file, f"Error in forecasting: {str(e)}", "Error Report")
        return None

def predict_compost_yield(df, report_file):
    """
    Regression analysis to predict compost yield from waste collection
    """
    try:
        # Prepare features and target
        features = ['lbs_collected', 'month', 'day_of_week']
        X = df[features]
        y = df['compost_created_lbs']
        
        # Split data ensuring time-based ordering
        train_size = int(len(df) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Train Random Forest model
        print("Training Random Forest model...")
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = rf_model.predict(X_test)

        # Calculate standard regression metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        explained_variance = explained_variance_score(y_test, y_pred)

        # Calculate additional metrics for comprehensive evaluation
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        # Write detailed metrics to report
        write_to_report(report_file, "\nRegression Model Performance Metrics:")
        write_to_report(report_file, f"Mean Squared Error (MSE): {mse:.4f} lbs²")
        write_to_report(report_file, f"Root Mean Squared Error (RMSE): {rmse:.4f} lbs")
        write_to_report(report_file, f"Mean Absolute Error (MAE): {mae:.4f} lbs")
        write_to_report(report_file, f"Mean Absolute Percentage Error (MAPE): {mape:.4f}%")
        write_to_report(report_file, f"R-squared Score: {r2:.4f}")
        write_to_report(report_file, f"Explained Variance Score: {explained_variance:.4f}")
        write_to_report(report_file, "Compost Yield Prediction Model", "Model Performance")
        write_to_report(report_file, f"Mean Squared Error: {mse:.2f}")
        write_to_report(report_file, f"Mean Absolute Error: {mae:.2f}")
        write_to_report(report_file, f"R-squared Score: {r2:.2f}\n")
        
       # Feature importance analysis
        importance_df = pd.DataFrame({
            'feature': features,
            'importance': rf_model.feature_importances_
        })
        importance_df = importance_df.sort_values('importance', ascending=False)

        # Calculate cumulative importance
        importance_df['cumulative_importance'] = importance_df['importance'].cumsum()

        write_to_report(report_file, "Feature Importance:")
        write_to_report(report_file, importance_df.to_string(index=False))
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        plt.bar(importance_df['feature'], importance_df['importance'])
        plt.title('Feature Importance in Compost Yield Prediction')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'))
        plt.close()
        
        return rf_model
    except Exception as e:
        write_to_report(report_file, f"Error in yield prediction: {str(e)}", "Error Report")
        return None

def classify_efficiency(df, report_file):
    try:
        write_to_report(report_file, "Collection Site Efficiency Analysis", "Site Analysis Results")
        
        # Calculate efficiency for each collection
        df['compost_efficiency'] = df['compost_created_lbs'] / df['lbs_collected']
        
        # Group by collection site
        efficiency_analysis = df.groupby('stop_name').agg({
            'compost_efficiency': ['mean', 'std', 'min', 'max', 'count']
        })
        
        # Flatten multi-index columns
        efficiency_analysis.columns = ['mean_efficiency', 'efficiency_std', 'min_efficiency', 'max_efficiency', 'collection_count']
        efficiency_analysis = efficiency_analysis.reset_index().rename(columns={'stop_name': 'collection_site'})
        
        # Calculate median efficiency for classification
        median_efficiency = efficiency_analysis['mean_efficiency'].median()
        
        # Classify sites as high or low efficiency
        efficiency_analysis['efficiency_level'] = efficiency_analysis['mean_efficiency'].apply(
            lambda x: 'High' if x >= median_efficiency else 'Low'
        )
        
        # Write overall statistics to report
        write_to_report(report_file, f"\nOverall Efficiency Statistics:")
        write_to_report(report_file, f"Average System Efficiency: {efficiency_analysis['mean_efficiency'].mean():.4f}")
        write_to_report(report_file, f"System Efficiency Range: {efficiency_analysis['min_efficiency'].min():.4f} - {efficiency_analysis['max_efficiency'].max():.4f}")
        
        # Write detailed site analysis to report
        write_to_report(report_file, f"\nDetailed Site Analysis:")
        for _, site in efficiency_analysis.iterrows():
            write_to_report(report_file, f"\nSite: {site['collection_site']}")
            write_to_report(report_file, f"Efficiency Level: {site['efficiency_level']}")
            write_to_report(report_file, f"Mean Efficiency: {site['mean_efficiency']:.4f} (±{site['efficiency_std']:.4f})")
            write_to_report(report_file, f"Efficiency Range: {site['min_efficiency']:.4f} - {site['max_efficiency']:.4f}")
            write_to_report(report_file, f"Number of Collections: {site['collection_count']}")
        
        # Calculate classification metrics
        y_true = (efficiency_analysis['mean_efficiency'] >= median_efficiency).astype(int)
        y_pred = (efficiency_analysis['efficiency_level'] == 'High').astype(int)
        
        # Calculate classification metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        confusion = confusion_matrix(y_true, y_pred)
        
        # Write classification metrics to report
        write_to_report(report_file, "\nEfficiency Classification Metrics:")
        write_to_report(report_file, f"Accuracy: {accuracy:.4f}")
        write_to_report(report_file, f"Precision: {precision:.4f}")
        write_to_report(report_file, f"Recall: {recall:.4f}")
        write_to_report(report_file, f"F1 Score: {f1:.4f}")
        write_to_report(report_file, f"\nConfusion Matrix:")
        write_to_report(report_file, f"True Negative: {confusion[0,0]}")
        write_to_report(report_file, f"False Positive: {confusion[0,1]}")
        write_to_report(report_file, f"False Negative: {confusion[1,0]}")
        write_to_report(report_file, f"True Positive: {confusion[1,1]}")
        
        # Create visualization of efficiency comparison
        plt.figure(figsize=(10, 6))
        sites = efficiency_analysis['collection_site']
        colors = ['green' if level == 'High' else 'orange' for level in efficiency_analysis['efficiency_level']]
        
        plt.bar(sites, 
               efficiency_analysis['mean_efficiency'],
               yerr=efficiency_analysis['efficiency_std'],
               capsize=5,
               color=colors)
        plt.axhline(y=median_efficiency, color='b', linestyle='--', label='Median Efficiency')
        plt.title('Collection Site Efficiency Comparison')
        plt.xlabel('Collection Site')
        plt.ylabel('Mean Efficiency (with std. dev.)')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'efficiency_comparison.png'))
        plt.close()
        
        return efficiency_analysis
    except Exception as e:
        write_to_report(report_file, f"Error in efficiency analysis: {str(e)}", "Error Report")
        return None

if __name__ == "__main__":
    print("Loading data...")
    df = load_and_clean_data()
    
    # Open report file
    with open(os.path.join(OUTPUT_DIR, 'predictive_analysis_report.txt'), 'w') as f:
        write_to_report(f, "Food Waste Predictive Analysis Report", "Executive Summary")
        write_to_report(f, "This report contains the results of three predictive analyses:")
        write_to_report(f, "1. Time Series Forecasting of Compost Production")
        write_to_report(f, "2. Compost Yield Prediction Model")
        write_to_report(f, "3. Collection Site Efficiency Classification")
        
        print("\nRunning time series forecast...")
        forecast = forecast_compost_output(df, f)
        
        print("\nRunning yield prediction...")
        model = predict_compost_yield(df, f)
        
        print("\nRunning efficiency classification...")
        efficiency_groups = classify_efficiency(df, f)
        
        write_to_report(f, "\nAnalysis completed successfully.", "Conclusion")
        print(f"\nAnalysis complete! Check {OUTPUT_DIR} directory for full results and generated plots.")
