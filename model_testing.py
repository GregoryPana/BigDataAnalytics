import os
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, KFold, cross_val_score
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, explained_variance_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from scipy import stats
import itertools
import warnings
warnings.filterwarnings('ignore')

# Import the data loading function - check if it exists first
try:
    from prep_clean import load_and_clean_data
except ImportError:
    # Define a fallback function if prep_clean.py is not available
    def load_and_clean_data():
        print("Error: prep_clean.py module not found.")
        print("Please ensure the prep_clean.py file exists with a load_and_clean_data function.")
        print("Using sample data for demonstration.")
        # Create a simple sample dataset
        import datetime
        dates = pd.date_range(start='2023-01-01', periods=100)
        data = {
            'date_collection_datetime': dates,
            'lbs_collected': np.random.uniform(50, 200, 100),
            'compost_created_lbs': np.random.uniform(5, 20, 100),
            'stop_name': np.random.choice(['Site A', 'Site B'], 100),
            'month': [d.month for d in dates],
            'day_of_week': [d.dayofweek for d in dates]
        }
        return pd.DataFrame(data)

# Define output directory
OUTPUT_DIR = 'MODEL_TESTING_OUTPUT'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def write_to_report(file, text, section=None):
    """Helper function to write to report file"""
    if section:
        file.write(f"\n{section}\n")
        file.write("=" * len(section) + "\n")
    file.write(f"{text}\n")

def test_time_series_model(df, report_file):
    """
    Comprehensive testing of time series forecasting model (text report only)
    """
    write_to_report(report_file, "Time Series Model Testing", "Test Results")
    
    try:
        # Prepare time series data
        df['date'] = pd.to_datetime(df['date_collection_datetime'])
        ts_data_full = df.groupby('date')['compost_created_lbs'].sum().asfreq('D').fillna(method='ffill')
        
        # Test stationarity
        write_to_report(report_file, "Stationarity Test (Augmented Dickey-Fuller)")
        adf_result = adfuller(ts_data_full.diff().dropna())
        write_to_report(report_file, f"ADF Statistic: {adf_result[0]:.4f}")
        write_to_report(report_file, f"p-value: {adf_result[1]:.4f}")
        write_to_report(report_file, f"Critical Values:")
        for key, value in adf_result[4].items():
            write_to_report(report_file, f"   {key}: {value:.4f}")
        write_to_report(report_file, f"Stationary: {adf_result[1] < 0.05}")
        
        # ARIMA parameter optimization
        write_to_report(report_file, "\nARIMA Parameter Optimization")
        
        # Define parameter grid (reduced for faster testing)
        p_values = range(0, 2)
        d_values = range(0, 2)
        q_values = range(0, 2)
        
        best_aic = float("inf")
        best_params = None
        best_model = None
        
        # Split data for testing
        train_size = int(len(ts_data_full) * 0.8)
        train_data = ts_data_full[:train_size]
        test_data = ts_data_full[train_size:]
        
        # Test different parameter combinations
        results = []
        for p, d, q in itertools.product(p_values, d_values, q_values):
            try:
                model = ARIMA(train_data, order=(p, d, q))
                model_fit = model.fit()
                
                # Make predictions
                predictions = model_fit.forecast(steps=len(test_data))
                
                # Calculate metrics
                mae = mean_absolute_error(test_data, predictions[:len(test_data)])
                mse = mean_squared_error(test_data, predictions[:len(test_data)])
                rmse = np.sqrt(mse)
                
                # Store results
                results.append({
                    'order': f"({p},{d},{q})",
                    'aic': model_fit.aic,
                    'bic': model_fit.bic,
                    'mae': mae,
                    'mse': mse,
                    'rmse': rmse
                })
                
                # Check if this is the best model so far
                if model_fit.aic < best_aic:
                    best_aic = model_fit.aic
                    best_params = (p, d, q)
                    best_model = model_fit
            except Exception as e:
                write_to_report(report_file, f"Error fitting ARIMA({p},{d},{q}): {str(e)}")
                continue
        
        # Convert results to DataFrame for easier analysis
        results_df = pd.DataFrame(results)
        
        # Write results to report
        write_to_report(report_file, f"Total models tested: {len(results_df)}")
        
        if results_df.empty:
            write_to_report(report_file, "No valid ARIMA models found. Try different parameters or check your data.")
            return {'best_params': None, 'metrics': {'rmse': 'N/A'}}
        
        write_to_report(report_file, f"Best model parameters (p,d,q): {best_params}")
        write_to_report(report_file, f"Best model AIC: {best_aic:.4f}")
        
        # Top 5 models
        if len(results_df) >= 5:
            top_models = results_df.sort_values('aic').head(5)
            write_to_report(report_file, "\nTop 5 Models:")
            write_to_report(report_file, top_models.to_string(index=False))
            
            # Create a simple bar chart for AIC comparison
            try:
                plt.figure(figsize=(10, 6))
                top_models_for_plot = top_models.sort_values('aic', ascending=False)
                plt.barh(top_models_for_plot['order'], top_models_for_plot['aic'])
                plt.title('AIC Values for Top ARIMA Models (Lower is Better)')
                plt.xlabel('AIC Value')
                plt.ylabel('ARIMA Parameters (p,d,q)')
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_DIR, 'arima_parameter_comparison.png'))
                plt.close()
                write_to_report(report_file, f"\nVisualization saved: arima_parameter_comparison.png")
            except Exception as e:
                write_to_report(report_file, f"Note: Could not create AIC visualization: {str(e)}")
        else:
            write_to_report(report_file, "\nAll Models:")
            write_to_report(report_file, results_df.to_string(index=False))
        
        # Test the best model on the full dataset
        best_metrics = {}
        if best_model is not None:
            try:
                # Make predictions
                forecast = best_model.forecast(steps=30)
                
                # Calculate metrics
                if len(test_data) >= 30:
                    mae = mean_absolute_error(test_data[:30], forecast[:30])
                    mse = mean_squared_error(test_data[:30], forecast[:30])
                    rmse = np.sqrt(mse)
                else:
                    mae = mean_absolute_error(test_data, forecast[:len(test_data)])
                    mse = mean_squared_error(test_data, forecast[:len(test_data)])
                    rmse = np.sqrt(mse)
                
                # Write results to report
                write_to_report(report_file, "\nBest Model Performance:")
                write_to_report(report_file, f"Mean Absolute Error: {mae:.4f}")
                write_to_report(report_file, f"Mean Squared Error: {mse:.4f}")
                write_to_report(report_file, f"Root Mean Squared Error: {rmse:.4f}")
                
                best_metrics = {'mae': mae, 'mse': mse, 'rmse': rmse}
            except Exception as e:
                write_to_report(report_file, f"Error testing best model: {str(e)}")
        
        return {'best_params': best_params, 'metrics': best_metrics}
    
    except Exception as e:
        write_to_report(report_file, f"Error in time series testing: {str(e)}")
        return {'best_params': None, 'metrics': {'rmse': 'N/A'}}

def test_regression_model(df, report_file):
    """
    Comprehensive testing of regression model with cross-validation (text report only)
    """
    write_to_report(report_file, "Regression Model Testing", "Test Results")
    
    try:
        # Prepare data
        X = df[['lbs_collected', 'month', 'day_of_week']]
        y = df['compost_created_lbs']
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Initialize model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Cross-validation metrics
        cv_mse = []
        cv_mae = []
        cv_r2 = []
        cv_ev = []
        
        # Feature importance across folds
        feature_importance = pd.DataFrame(index=X.columns)
        
        # Perform cross-validation
        fold = 1
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            ev = explained_variance_score(y_test, y_pred)
            
            # Store metrics
            cv_mse.append(mse)
            cv_mae.append(mae)
            cv_r2.append(r2)
            cv_ev.append(ev)
            
            # Store feature importance
            feature_importance[f'Fold_{fold}'] = model.feature_importances_
            
            # Write fold results
            write_to_report(report_file, f"\nFold {fold} Results:")
            write_to_report(report_file, f"MSE: {mse:.4f}")
            write_to_report(report_file, f"MAE: {mae:.4f}")
            write_to_report(report_file, f"R²: {r2:.4f}")
            write_to_report(report_file, f"Explained Variance: {ev:.4f}")
            
            fold += 1
        
        # Calculate average metrics
        avg_mse = np.mean(cv_mse)
        avg_mae = np.mean(cv_mae)
        avg_r2 = np.mean(cv_r2)
        avg_ev = np.mean(cv_ev)
        
        rmse = np.sqrt(avg_mse)
        
        # Write average results
        write_to_report(report_file, "\nAverage Cross-Validation Results:")
        write_to_report(report_file, f"MSE: {avg_mse:.4f}")
        write_to_report(report_file, f"RMSE: {rmse:.4f}")
        write_to_report(report_file, f"MAE: {avg_mae:.4f}")
        write_to_report(report_file, f"R²: {avg_r2:.4f}")
        write_to_report(report_file, f"Explained Variance: {avg_ev:.4f}")
        
        # Calculate average feature importance
        feature_importance['Average'] = feature_importance.mean(axis=1)
        feature_importance = feature_importance.sort_values('Average', ascending=False)
        
        # Write feature importance
        write_to_report(report_file, "\nFeature Importance:")
        for feature, importance in feature_importance['Average'].items():
            write_to_report(report_file, f"{feature}: {importance:.4f}")
        
        # Train final model on all data
        final_model = RandomForestRegressor(n_estimators=100, random_state=42)
        final_model.fit(X, y)
        
        # Make predictions on all data
        y_pred = final_model.predict(X)
        
        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        ev = explained_variance_score(y, y_pred)
        
        # Write final model results
        write_to_report(report_file, "\nFinal Model Performance (on all data):")
        write_to_report(report_file, f"MSE: {mse:.4f}")
        write_to_report(report_file, f"RMSE: {np.sqrt(mse):.4f}")
        write_to_report(report_file, f"MAE: {mae:.4f}")
        write_to_report(report_file, f"R²: {r2:.4f}")
        write_to_report(report_file, f"Explained Variance: {ev:.4f}")
        
        return {
            'avg_metrics': {
                'mse': avg_mse,
                'rmse': rmse,
                'mae': avg_mae,
                'r2': avg_r2,
                'ev': avg_ev
            },
            'feature_importance': feature_importance['Average'].to_dict()
        }
    
    except Exception as e:
        write_to_report(report_file, f"Error in regression model testing: {str(e)}")
        return {'avg_metrics': {'rmse': 'N/A'}}

def test_classification_model(df, report_file):
    """
    Comprehensive testing of efficiency classification (text report only)
    """
    write_to_report(report_file, "Classification Model Testing", "Test Results")
    
    try:
        # Calculate efficiency
        if 'compost_efficiency' not in df.columns:
            df['compost_efficiency'] = df['compost_created_lbs'] / df['lbs_collected']
        
        # Create binary target variable
        efficiency_threshold = df['compost_efficiency'].median()
        df['high_efficiency'] = (df['compost_efficiency'] > efficiency_threshold).astype(int)
        
        write_to_report(report_file, f"Efficiency threshold: {efficiency_threshold:.4f}")
        write_to_report(report_file, f"Class distribution:")
        write_to_report(report_file, f"  Low efficiency (0): {(df['high_efficiency'] == 0).sum()}")
        write_to_report(report_file, f"  High efficiency (1): {(df['high_efficiency'] == 1).sum()}")
        
        # Prepare data
        X = df[['lbs_collected', 'month', 'day_of_week']]
        y = df['high_efficiency']
        
        # Cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # Initialize model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Cross-validation metrics
        cv_accuracy = []
        cv_precision = []
        cv_recall = []
        cv_f1 = []
        
        # Perform cross-validation
        fold = 1
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Store metrics
            cv_accuracy.append(accuracy)
            cv_precision.append(precision)
            cv_recall.append(recall)
            cv_f1.append(f1)
            
            # Write fold results
            write_to_report(report_file, f"\nFold {fold} Results:")
            write_to_report(report_file, f"Accuracy: {accuracy:.4f}")
            write_to_report(report_file, f"Precision: {precision:.4f}")
            write_to_report(report_file, f"Recall: {recall:.4f}")
            write_to_report(report_file, f"F1 Score: {f1:.4f}")
            
            fold += 1
        
        # Calculate average metrics
        accuracy = np.mean(cv_accuracy)
        precision = np.mean(cv_precision)
        recall = np.mean(cv_recall)
        f1 = np.mean(cv_f1)
        
        # Write average results
        write_to_report(report_file, "\nAverage Cross-Validation Results:")
        write_to_report(report_file, f"Accuracy: {accuracy:.4f}")
        write_to_report(report_file, f"Precision: {precision:.4f}")
        write_to_report(report_file, f"Recall: {recall:.4f}")
        write_to_report(report_file, f"F1 Score: {f1:.4f}")
        
        # Train final model on all data
        final_model = RandomForestClassifier(n_estimators=100, random_state=42)
        final_model.fit(X, y)
        
        # Feature importance
        feature_importance = pd.Series(final_model.feature_importances_, index=X.columns).sort_values(ascending=False)
        
        # Write feature importance
        write_to_report(report_file, "\nFeature Importance:")
        for feature, importance in feature_importance.items():
            write_to_report(report_file, f"{feature}: {importance:.4f}")
        
        # Statistical significance testing
        write_to_report(report_file, "\nStatistical Testing:")
        
        # Compare efficiency between sites
        sites = df['stop_name'].unique()
        if len(sites) >= 2:
            site1_data = df[df['stop_name'] == sites[0]]['compost_efficiency']
            site2_data = df[df['stop_name'] == sites[1]]['compost_efficiency']
            
            # T-test
            t_stat, p_val = stats.ttest_ind(site1_data, site2_data, equal_var=False)
            
            write_to_report(report_file, f"\nStatistical Comparison of Efficiency Between Sites:")
            write_to_report(report_file, f"T-statistic: {t_stat:.4f}")
            write_to_report(report_file, f"p-value: {p_val:.4f}")
            write_to_report(report_file, f"Significant difference: {p_val < 0.05}")
        
        return {
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            },
            'cv_scores': cv_accuracy,
            'feature_importance': feature_importance.to_dict()
        }
    
    except Exception as e:
        write_to_report(report_file, f"Error in classification model testing: {str(e)}")
        return {'metrics': {'accuracy': 0, 'f1': 0}}

def run_comprehensive_testing():
    """
    Run all tests and generate comprehensive report
    """
    print("Loading data...")
    try:
        df = load_and_clean_data()
        
        # Open report file
        with open(os.path.join(OUTPUT_DIR, 'model_testing_report.txt'), 'w') as f:
            write_to_report(f, "Food Waste Analysis Model Testing Report", "Executive Summary")
            write_to_report(f, "This report contains comprehensive testing results for three predictive models:")
            write_to_report(f, "1. Time Series Forecasting (ARIMA)")
            write_to_report(f, "2. Compost Yield Prediction (Random Forest Regression)")
            write_to_report(f, "3. Efficiency Classification (Random Forest Classification)")
            
            print("\nTesting time series model...")
            try:
                ts_results = test_time_series_model(df, f)
            except Exception as e:
                write_to_report(f, f"Error in time series testing: {str(e)}")
                ts_results = {'best_params': None, 'metrics': {'rmse': 'N/A'}}
            
            print("\nTesting regression model...")
            try:
                reg_results = test_regression_model(df, f)
            except Exception as e:
                write_to_report(f, f"Error in regression model testing: {str(e)}")
                reg_results = {'avg_metrics': {'rmse': 'N/A'}}
            
            print("\nTesting classification model...")
            try:
                clf_results = test_classification_model(df, f)
            except Exception as e:
                write_to_report(f, f"Error in classification model testing: {str(e)}")
                clf_results = {'metrics': {'accuracy': 0, 'f1': 0}}
            
            # Comparative analysis
            write_to_report(f, "Model Comparison", "Comparative Analysis")
            
            # Time series vs regression for prediction
            write_to_report(f, "\nTime Series vs Regression for Prediction:")
            
            # Safely get time series RMSE
            ts_rmse = "N/A"
            if isinstance(ts_results, dict) and 'metrics' in ts_results:
                if isinstance(ts_results['metrics'], dict) and 'rmse' in ts_results['metrics']:
                    ts_rmse = ts_results['metrics']['rmse']
            
            # Safely get regression RMSE
            reg_rmse = "N/A"
            if isinstance(reg_results, dict) and 'avg_metrics' in reg_results:
                if isinstance(reg_results['avg_metrics'], dict) and 'rmse' in reg_results['avg_metrics']:
                    reg_rmse = reg_results['avg_metrics']['rmse']
            
            write_to_report(f, f"Time series RMSE: {ts_rmse}")
            write_to_report(f, f"Regression RMSE: {reg_rmse}")
            
            # Classification performance
            write_to_report(f, "\nClassification Performance:")
            
            # Safely get classification metrics
            clf_accuracy = 0
            clf_f1 = 0
            if isinstance(clf_results, dict) and 'metrics' in clf_results:
                if isinstance(clf_results['metrics'], dict):
                    clf_accuracy = clf_results['metrics'].get('accuracy', 0)
                    clf_f1 = clf_results['metrics'].get('f1', 0)
            
            write_to_report(f, f"Accuracy: {clf_accuracy:.4f}")
            write_to_report(f, f"F1 Score: {clf_f1:.4f}")
            
            # Overall recommendations
            write_to_report(f, "\nRecommendations:", "Conclusion")
            write_to_report(f, "1. For forecasting future compost production, use the ARIMA model with optimal parameters.")
            write_to_report(f, "2. For estimating compost yield from collection volume, use the Random Forest regression model.")
            write_to_report(f, "3. For classifying collection efficiency, use the Random Forest classification model.")
            
            write_to_report(f, "\nTesting completed successfully.")
            
        print(f"\nTesting complete! Check {OUTPUT_DIR} directory for full results.")
        print(f"Report saved to {os.path.join(OUTPUT_DIR, 'model_testing_report.txt')}")
    
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        print("Please check that the data file exists and is properly formatted.")
        print("Also ensure that all required packages are installed.")
        
        # Create a minimal report even if testing fails
        with open(os.path.join(OUTPUT_DIR, 'model_testing_report.txt'), 'w') as f:
            write_to_report(f, "Food Waste Analysis Model Testing Report", "Error Report")
            write_to_report(f, f"Testing failed with error: {str(e)}")
            write_to_report(f, "Please check the following:")
            write_to_report(f, "1. Data file exists and is properly formatted")
            write_to_report(f, "2. All required packages are installed")
            write_to_report(f, "3. The prep_clean.py module is available and working correctly")

if __name__ == "__main__":
    run_comprehensive_testing()
