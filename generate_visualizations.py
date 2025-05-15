import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')
import re
from datetime import datetime

# Define directories
OUTPUT_DIR = 'MODEL_TESTING_OUTPUT'
EVAL_DIR = 'EVALUATION_REPORT'

# Create evaluation directory if it doesn't exist
if not os.path.exists(EVAL_DIR):
    os.makedirs(EVAL_DIR)

def extract_section(content, section_name, end_marker='\n\n'):
    """Extract a section from the report content"""
    pattern = f"{section_name}.*?(?={end_marker})"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        return match.group(0)
    return None

def extract_metrics(content, metric_name):
    """Extract metrics from the report content"""
    pattern = f"{metric_name}: (\\d+\\.\\d+)"
    match = re.search(pattern, content)
    if match:
        return float(match.group(1))
    return None

def copy_image(src_file, dest_file):
    """Copy an image file to the evaluation directory"""
    if os.path.exists(src_file):
        import shutil
        shutil.copy2(src_file, dest_file)
        return True
    return False

def generate_evaluation_report():
    """Generate a comprehensive evaluation report"""
    print("Generating comprehensive evaluation report...")
    
    # Check if model testing report exists
    report_file = os.path.join(OUTPUT_DIR, 'model_testing_report.txt')
    if not os.path.exists(report_file):
        print(f"Model testing report not found: {report_file}")
        print("Please run model_testing.py first.")
        return
    
    # Read the model testing report
    with open(report_file, 'r') as f:
        content = f.read()
    
    # Create the evaluation report file
    eval_report_file = os.path.join(EVAL_DIR, 'evaluation_report.html')
    
    # Start building the HTML report
    html = []
    html.append('<!DOCTYPE html>')
    html.append('<html lang="en">')
    html.append('<head>')
    html.append('    <meta charset="UTF-8">')
    html.append('    <meta name="viewport" content="width=device-width, initial-scale=1.0">')
    html.append('    <title>Food Waste Analysis - Model Evaluation Report</title>')
    html.append('    <style>')
    html.append('        body { font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; color: #333; }')
    html.append('        h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }')
    html.append('        h2 { color: #2980b9; margin-top: 30px; border-bottom: 1px solid #ddd; padding-bottom: 5px; }')
    html.append('        h3 { color: #3498db; }')
    html.append('        .container { max-width: 1200px; margin: 0 auto; }')
    html.append('        .metric { background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }')
    html.append('        .metric h3 { margin-top: 0; }')
    html.append('        .metric-value { font-weight: bold; color: #2c3e50; }')
    html.append('        .visualization { margin: 20px 0; text-align: center; }')
    html.append('        .visualization img { max-width: 100%; border: 1px solid #ddd; border-radius: 5px; }')
    html.append('        .interpretation { background-color: #e8f4f8; padding: 15px; border-left: 4px solid #3498db; margin: 20px 0; }')
    html.append('        table { border-collapse: collapse; width: 100%; margin: 20px 0; }')
    html.append('        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }')
    html.append('        th { background-color: #f2f2f2; }')
    html.append('        tr:nth-child(even) { background-color: #f9f9f9; }')
    html.append('        .conclusion { background-color: #f0f7fb; padding: 20px; border-radius: 5px; margin-top: 30px; }')
    html.append('        .footer { margin-top: 50px; text-align: center; font-size: 0.9em; color: #7f8c8d; }')
    html.append('    </style>')
    html.append('</head>')
    html.append('<body>')
    html.append('    <div class="container">')
    
    # Header
    html.append('        <h1>Food Waste Analysis - Model Evaluation Report</h1>')
    html.append(f'        <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>')
    
    # Introduction
    html.append('        <div class="interpretation">')
    html.append('            <p>This report presents a comprehensive evaluation of the predictive models developed for the food waste analysis project. The evaluation includes time series forecasting, regression analysis for yield prediction, and classification for efficiency categorization. The report includes performance metrics, visualizations, and interpretations to support decision-making and planning.</p>')
    html.append('        </div>')
    
    # Time Series Model Evaluation
    html.append('        <h2>1. Time Series Forecasting Evaluation</h2>')
    
    # Extract time series metrics
    ts_section = extract_section(content, 'Time Series Model Testing', 'Regression Model Testing')
    if ts_section:
        # Extract metrics
        mae = extract_metrics(ts_section, 'Mean Absolute Error')
        mse = extract_metrics(ts_section, 'Mean Squared Error')
        rmse = extract_metrics(ts_section, 'Root Mean Squared Error')
        
        # Display metrics
        html.append('        <div class="metric">')
        html.append('            <h3>Performance Metrics</h3>')
        html.append('            <table>')
        html.append('                <tr><th>Metric</th><th>Value</th><th>Interpretation</th></tr>')
        if mae:
            html.append(f'                <tr><td>Mean Absolute Error (MAE)</td><td class="metric-value">{mae:.4f}</td><td>Average absolute difference between predicted and actual values in pounds.</td></tr>')
        if mse:
            html.append(f'                <tr><td>Mean Squared Error (MSE)</td><td class="metric-value">{mse:.4f}</td><td>Average squared difference between predicted and actual values, emphasizing larger errors.</td></tr>')
        if rmse:
            html.append(f'                <tr><td>Root Mean Squared Error (RMSE)</td><td class="metric-value">{rmse:.4f}</td><td>Square root of MSE, providing an error measure in the same units as the data (pounds).</td></tr>')
        html.append('            </table>')
        html.append('        </div>')
    
    # ARIMA Parameter Comparison
    html.append('        <div class="visualization">')
    html.append('            <h3>ARIMA Parameter Comparison</h3>')
    arima_img_src = os.path.join(OUTPUT_DIR, 'arima_parameter_comparison.png')
    arima_img_dest = os.path.join(EVAL_DIR, 'arima_parameter_comparison.png')
    if copy_image(arima_img_src, arima_img_dest):
        html.append('            <img src="arima_parameter_comparison.png" alt="ARIMA Parameter Comparison">')
        html.append('            <p>Comparison of AIC values for different ARIMA model parameters. Lower AIC values indicate better model fit.</p>')
    else:
        html.append('            <p>ARIMA parameter comparison visualization not available.</p>')
    html.append('        </div>')
    
    # Forecast Visualization
    html.append('        <div class="visualization">')
    html.append('            <h3>Forecast vs Actual Values</h3>')
    forecast_img_src = os.path.join(OUTPUT_DIR, 'forecast_vs_actual.png')
    forecast_img_dest = os.path.join(EVAL_DIR, 'forecast_vs_actual.png')
    if copy_image(forecast_img_src, forecast_img_dest):
        html.append('            <img src="forecast_vs_actual.png" alt="Forecast vs Actual Values">')
        html.append('            <p>Comparison of forecasted values against actual values over time, showing the model\'s predictive accuracy.</p>')
    else:
        html.append('            <p>Forecast visualization not available.</p>')
    html.append('        </div>')
    
    # Time Series Interpretation
    html.append('        <div class="interpretation">')
    html.append('            <h3>Interpretation</h3>')
    html.append('            <p>The time series forecasting model provides predictions for future compost production based on historical patterns. The model\'s performance metrics indicate the accuracy of these predictions, with lower values suggesting better forecasting capability. These forecasts can be used for production capacity planning and resource allocation.</p>')
    if rmse:
        html.append(f'            <p>With an RMSE of {rmse:.4f} pounds, stakeholders can expect predictions to be within approximately ±{rmse*2:.4f} pounds of actual values with 95% confidence. This level of accuracy is {("sufficient" if rmse < 50 else "moderate" if rmse < 100 else "challenging")} for operational planning.</p>')
    html.append('        </div>')
    
    # Regression Model Evaluation
    html.append('        <h2>2. Yield Prediction Model Evaluation</h2>')
    
    # Extract regression metrics
    reg_section = extract_section(content, 'Regression Model Testing', 'Classification Model Testing')
    if reg_section:
        # Extract metrics
        r2 = extract_metrics(reg_section, 'R-squared')
        reg_mae = extract_metrics(reg_section, 'Mean Absolute Error')
        reg_mse = extract_metrics(reg_section, 'Mean Squared Error')
        
        # Display metrics
        html.append('        <div class="metric">')
        html.append('            <h3>Performance Metrics</h3>')
        html.append('            <table>')
        html.append('                <tr><th>Metric</th><th>Value</th><th>Interpretation</th></tr>')
        if r2:
            html.append(f'                <tr><td>R-squared (R²)</td><td class="metric-value">{r2:.4f}</td><td>Proportion of variance in the dependent variable explained by the model. Values closer to 1 indicate better fit.</td></tr>')
        if reg_mae:
            html.append(f'                <tr><td>Mean Absolute Error (MAE)</td><td class="metric-value">{reg_mae:.4f}</td><td>Average absolute difference between predicted and actual yield values.</td></tr>')
        if reg_mse:
            html.append(f'                <tr><td>Mean Squared Error (MSE)</td><td class="metric-value">{reg_mse:.4f}</td><td>Average squared difference between predicted and actual yield values.</td></tr>')
        html.append('            </table>')
        html.append('        </div>')
    
    # Regression Feature Importance
    html.append('        <div class="visualization">')
    html.append('            <h3>Feature Importance for Yield Prediction</h3>')
    reg_img_src = os.path.join(OUTPUT_DIR, 'regression_feature_importance.png')
    reg_img_dest = os.path.join(EVAL_DIR, 'regression_feature_importance.png')
    if copy_image(reg_img_src, reg_img_dest):
        html.append('            <img src="regression_feature_importance.png" alt="Regression Feature Importance">')
        html.append('            <p>Relative importance of different features in predicting compost yield, helping identify the key factors influencing production.</p>')
    else:
        html.append('            <p>Regression feature importance visualization not available.</p>')
    html.append('        </div>')
    
    # Regression Interpretation
    html.append('        <div class="interpretation">')
    html.append('            <h3>Interpretation</h3>')
    html.append('            <p>The regression model predicts compost yield based on various input features. The feature importance visualization highlights which factors have the most significant impact on yield, providing insights for optimization.</p>')
    if r2:
        r2_interpretation = "excellent" if r2 > 0.8 else "good" if r2 > 0.6 else "moderate" if r2 > 0.4 else "weak"
        html.append(f'            <p>With an R² value of {r2:.4f}, the model demonstrates {r2_interpretation} predictive power, explaining approximately {r2*100:.1f}% of the variance in compost yield. This level of accuracy can {("strongly" if r2 > 0.7 else "moderately" if r2 > 0.5 else "somewhat")} support collection strategy optimization.</p>')
    html.append('        </div>')
    
    # Classification Model Evaluation
    html.append('        <h2>3. Efficiency Classification Model Evaluation</h2>')
    
    # Extract classification metrics
    class_section = extract_section(content, 'Classification Model Testing', 'end')
    if class_section:
        # Extract metrics
        accuracy = extract_metrics(class_section, 'Accuracy')
        precision = extract_metrics(class_section, 'Precision')
        recall = extract_metrics(class_section, 'Recall')
        f1 = extract_metrics(class_section, 'F1 Score')
        
        # Display metrics
        html.append('        <div class="metric">')
        html.append('            <h3>Performance Metrics</h3>')
        html.append('            <table>')
        html.append('                <tr><th>Metric</th><th>Value</th><th>Interpretation</th></tr>')
        if accuracy:
            html.append(f'                <tr><td>Accuracy</td><td class="metric-value">{accuracy:.4f}</td><td>Proportion of correctly classified instances. Values closer to 1 indicate better performance.</td></tr>')
        if precision:
            html.append(f'                <tr><td>Precision</td><td class="metric-value">{precision:.4f}</td><td>Proportion of true positives among instances predicted as positive. Measures model\'s ability to avoid false positives.</td></tr>')
        if recall:
            html.append(f'                <tr><td>Recall</td><td class="metric-value">{recall:.4f}</td><td>Proportion of true positives among actual positive instances. Measures model\'s ability to find all positive instances.</td></tr>')
        if f1:
            html.append(f'                <tr><td>F1 Score</td><td class="metric-value">{f1:.4f}</td><td>Harmonic mean of precision and recall. Balances the trade-off between precision and recall.</td></tr>')
        html.append('            </table>')
        html.append('        </div>')
    
    # Classification Confusion Matrix
    html.append('        <div class="visualization">')
    html.append('            <h3>Classification Confusion Matrix</h3>')
    cm_img_src = os.path.join(OUTPUT_DIR, 'classification_confusion_matrix.png')
    cm_img_dest = os.path.join(EVAL_DIR, 'classification_confusion_matrix.png')
    if copy_image(cm_img_src, cm_img_dest):
        html.append('            <img src="classification_confusion_matrix.png" alt="Classification Confusion Matrix">')
        html.append('            <p>Visualization of model predictions vs. actual classes, showing true positives, false positives, true negatives, and false negatives.</p>')
    else:
        html.append('            <p>Classification confusion matrix visualization not available.</p>')
    html.append('        </div>')
    
    # Classification Feature Importance
    html.append('        <div class="visualization">')
    html.append('            <h3>Feature Importance for Classification</h3>')
    class_img_src = os.path.join(OUTPUT_DIR, 'classification_feature_importance.png')
    class_img_dest = os.path.join(EVAL_DIR, 'classification_feature_importance.png')
    if copy_image(class_img_src, class_img_dest):
        html.append('            <img src="classification_feature_importance.png" alt="Classification Feature Importance">')
        html.append('            <p>Relative importance of different features in classifying collection site efficiency, helping identify key factors for site performance.</p>')
    else:
        html.append('            <p>Classification feature importance visualization not available.</p>')
    html.append('        </div>')
    
    # Classification Interpretation
    html.append('        <div class="interpretation">')
    html.append('            <h3>Interpretation</h3>')
    html.append('            <p>The classification model categorizes collection sites based on their efficiency. This categorization helps identify high-performing and underperforming sites, enabling targeted improvement strategies.</p>')
    if accuracy and f1:
        acc_interpretation = "excellent" if accuracy > 0.9 else "good" if accuracy > 0.8 else "moderate" if accuracy > 0.7 else "fair"
        html.append(f'            <p>With an accuracy of {accuracy:.4f} and F1 score of {f1:.4f}, the model demonstrates {acc_interpretation} performance in classifying site efficiency. This level of accuracy can {("strongly" if accuracy > 0.85 else "moderately" if accuracy > 0.75 else "somewhat")} support performance improvement targeting.</p>')
    html.append('        </div>')
    
    # Overall Conclusions
    html.append('        <h2>4. Overall Conclusions and Recommendations</h2>')
    html.append('        <div class="conclusion">')
    html.append('            <h3>Summary of Model Performance</h3>')
    html.append('            <p>The predictive models developed for the food waste analysis project demonstrate the ability to forecast compost production, predict yield based on various factors, and classify collection sites by efficiency. These models provide valuable insights for operational planning and optimization.</p>')
    
    # Generate overall recommendations based on available metrics
    html.append('            <h3>Recommendations</h3>')
    html.append('            <ol>')
    html.append('                <li><strong>Production Planning:</strong> Use the time series forecasting model to anticipate future compost production levels and plan resource allocation accordingly.</li>')
    html.append('                <li><strong>Collection Strategy:</strong> Focus on the key features identified in the regression model to optimize compost yield from collection efforts.</li>')
    html.append('                <li><strong>Site Improvement:</strong> Target underperforming sites identified by the classification model for process improvements and best practice sharing.</li>')
    html.append('                <li><strong>Data Collection:</strong> Continue collecting high-quality data to refine and improve model performance over time.</li>')
    html.append('                <li><strong>Model Monitoring:</strong> Regularly evaluate model performance against new data to ensure continued accuracy and relevance.</li>')
    html.append('            </ol>')
    html.append('        </div>')
    
    # Next Steps
    html.append('        <h2>5. Next Steps</h2>')
    html.append('        <div class="interpretation">')
    html.append('            <p>To further enhance the predictive capabilities and operational impact of these models, consider the following next steps:</p>')
    html.append('            <ol>')
    html.append('                <li>Implement the models in a production environment with regular retraining schedules.</li>')
    html.append('                <li>Develop a dashboard for real-time monitoring of key performance indicators.</li>')
    html.append('                <li>Conduct A/B testing of operational changes suggested by the models.</li>')
    html.append('                <li>Explore advanced modeling techniques such as ensemble methods or deep learning for potentially improved performance.</li>')
    html.append('                <li>Integrate additional data sources such as weather patterns or seasonal events that may impact collection and processing.</li>')
    html.append('            </ol>')
    html.append('        </div>')
    
    # Footer
    html.append('        <div class="footer">')
    html.append('            <p>Food Waste Analysis Project - Model Evaluation Report</p>')
    html.append(f'            <p>Generated on {datetime.now().strftime("%Y-%m-%d")} using automated evaluation tools</p>')
    html.append('        </div>')
    
    html.append('    </div>')
    html.append('</body>')
    html.append('</html>')
    
    # Write the HTML report
    with open(eval_report_file, 'w') as f:
        f.write('\n'.join(html))
    
    print(f"Evaluation report generated: {eval_report_file}")
    print("Open this HTML file in a web browser to view the complete report.")

if __name__ == "__main__":
    generate_evaluation_report()