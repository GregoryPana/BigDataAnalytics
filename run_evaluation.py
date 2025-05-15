import os
import subprocess
import sys

def check_directory(directory, create=False):
    """Check if a directory exists and optionally create it"""
    if not os.path.exists(directory):
        if create:
            os.makedirs(directory)
            print(f"Created directory: {directory}")
            return True
        else:
            print(f"Directory not found: {directory}")
            return False
    return True

def run_evaluation():
    """Run the complete evaluation process"""
    # Check for required directories
    model_testing_output = 'MODEL_TESTING_OUTPUT'
    eval_dir = 'EVALUATION_REPORT'
    desc_dir = 'DESC_ANAL_OUTPUT'
    
    # Check if model testing output exists
    if not check_directory(model_testing_output):
        print("Model testing results not found.")
        print("Please run model_testing.py first.")
        return False
    
    # Check if model testing report exists
    report_file = os.path.join(model_testing_output, 'model_testing_report.txt')
    if not os.path.exists(report_file):
        print(f"Model testing report not found: {report_file}")
        print("Please run model_testing.py first.")
        return False
    
    # Check if descriptive analysis directory exists
    if not check_directory(desc_dir):
        print("Descriptive analysis outputs not found.")
        print("Please run descriptive_analysis.py first or create the directory manually.")
        print("Note: The report will still be generated but without descriptive analysis visualizations.")
    
    # Create evaluation directory if it doesn't exist
    check_directory(eval_dir, create=True)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    try:
        subprocess.run([sys.executable, 'generate_visualizations.py'], check=True)
        print("Visualizations generated successfully.")
    except subprocess.CalledProcessError:
        print("Error generating visualizations.")
        return False
    
    # Generate evaluation report
    print("\nGenerating evaluation report...")
    try:
        subprocess.run([sys.executable, 'generate_evaluation_report.py'], check=True)
        print("Evaluation report generated successfully.")
    except subprocess.CalledProcessError:
        print("Error generating evaluation report.")
        return False
    
    print("\nEvaluation process completed successfully.")
    print(f"Open {os.path.join(eval_dir, 'evaluation_report.html')} in a web browser to view the report.")
    return True

if __name__ == "__main__":
    run_evaluation()
