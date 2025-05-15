import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from prep_clean import load_and_clean_data, save_processed_data
import calendar

# Define output directory
OUTPUT_DIR = 'DESC_ANAL_OUTPUT'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def write_to_report(file, text, section=None):
    """Helper function to write to report file"""
    if section:
        file.write(f"\n{section}\n")
        file.write("=" * len(section) + "\n")
    file.write(f"{text}\n")

def analyze_collection_volume(df, report_file):
    """
    Analyze food waste collection volume and efficiency
    """
    try:
        # Monthly collection statistics
        monthly_stats = df.groupby('month').agg({
            'lbs_collected': ['mean', 'sum', 'count'],
            'compost_created_lbs': ['mean', 'sum'],
            'compost_efficiency': 'mean'
        }).round(2)
        
        # Add month names
        month_names = {i: calendar.month_name[i] for i in range(1, 13)}
        monthly_stats.index = monthly_stats.index.map(lambda x: month_names[x])
        
        write_to_report(report_file, "\nMonthly Collection Statistics:", "Collection Volume Analysis")
        write_to_report(report_file, monthly_stats.to_string())
        
        # Plot monthly trends
        plt.figure(figsize=(12, 6))
        monthly_avg = df.groupby('month')['lbs_collected'].mean()
        monthly_avg.index = monthly_avg.index.map(lambda x: month_names[x])
        monthly_avg.plot(kind='bar')
        plt.title('Average Monthly Food Waste Collection')
        plt.xlabel('Month')
        plt.ylabel('Average Pounds Collected')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'monthly_collection_trend.png'))
        plt.close()
        
        # Daily collection patterns
        daily_stats = df.groupby('day_of_week').agg({
            'lbs_collected': ['mean', 'count'],
            'compost_efficiency': 'mean'
        }).round(2)
        
        # Add day names
        day_names = {i: calendar.day_name[i] for i in range(7)}
        daily_stats.index = daily_stats.index.map(lambda x: day_names[x])
        
        write_to_report(report_file, "\nDaily Collection Statistics:", "Daily Patterns")
        write_to_report(report_file, daily_stats.to_string())
        
        # Plot daily trends
        plt.figure(figsize=(10, 6))
        daily_avg = df.groupby('day_of_week')['lbs_collected'].mean()
        daily_avg.index = daily_avg.index.map(lambda x: day_names[x])
        daily_avg.plot(kind='bar')
        plt.title('Average Daily Food Waste Collection')
        plt.xlabel('Day of Week')
        plt.ylabel('Average Pounds Collected')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'daily_collection_trend.png'))
        plt.close()
        
    except Exception as e:
        write_to_report(report_file, f"Error in collection volume analysis: {str(e)}", "Error Report")

def analyze_compost_efficiency(df, report_file):
    """
    Analyze compost conversion efficiency and identify high-yield locations
    """
    # Calculate efficiency metrics by location
    efficiency_stats = df.groupby('stop_name').agg({
        'lbs_collected': ['sum', 'mean', 'count'],
        'compost_created_lbs': ['sum', 'mean'],
        'compost_efficiency': ['mean', 'std']
    }).round(3)
    
    efficiency_stats = efficiency_stats.sort_values(('compost_efficiency', 'mean'), ascending=False)
    
    write_to_report(report_file, "\nCompost Efficiency by Collection Site:", "Efficiency Analysis")
    write_to_report(report_file, efficiency_stats.to_string())
    
    # Identify high-yield locations
    high_yield_threshold = df['lbs_collected'].mean() + df['lbs_collected'].std()
    high_yield_sites = df[df['lbs_collected'] > high_yield_threshold]['stop_name'].unique()
    
    write_to_report(report_file, "\nHigh-Yield Locations:")
    write_to_report(report_file, f"Threshold for high-yield classification: {high_yield_threshold:.2f} lbs")
    write_to_report(report_file, f"Number of high-yield sites: {len(high_yield_sites)}")
    for site in high_yield_sites:
        site_stats = df[df['stop_name'] == site].agg({
            'lbs_collected': ['mean', 'sum'],
            'compost_efficiency': 'mean'
        }).round(2)
        write_to_report(report_file, f"\n{site}:")
        write_to_report(report_file, f"  Average Collection: {site_stats['lbs_collected']['mean']:.2f} lbs")
        write_to_report(report_file, f"  Total Collection: {site_stats['lbs_collected']['sum']:.2f} lbs")
        write_to_report(report_file, f"  Efficiency: {site_stats['compost_efficiency']['mean']:.2%}")
    
    # Plot efficiency distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='compost_efficiency', bins=30)
    plt.axvline(df['compost_efficiency'].mean(), color='r', linestyle='--', label='Mean Efficiency')
    plt.title('Distribution of Compost Efficiency')
    plt.xlabel('Efficiency (Compost Created / Waste Collected)')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'efficiency_distribution.png'))
    plt.close()
    
    # Efficiency vs Collection Volume scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df['lbs_collected'], df['compost_efficiency'], alpha=0.5)
    plt.axvline(high_yield_threshold, color='r', linestyle='--', label='High-Yield Threshold')
    plt.title('Efficiency vs Collection Volume')
    plt.xlabel('Pounds Collected')
    plt.ylabel('Compost Efficiency')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'efficiency_vs_volume.png'))
    plt.close()

def analyze_time_trends(df, report_file):
    """
    Analyze temporal trends in waste collection and composting
    """
    # Monthly trends over time
    time_series = df.set_index('date_collection_datetime')
    monthly_series = time_series.resample('M').agg({
        'lbs_collected': 'sum',
        'compost_created_lbs': 'sum',
        'compost_efficiency': 'mean'
    })
    
    write_to_report(report_file, "\nMonthly Time Series Analysis:", "Temporal Trends")
    write_to_report(report_file, monthly_series.to_string())
    
    # Calculate growth rates
    year_stats = df.groupby('year').agg({
        'lbs_collected': 'sum',
        'compost_created_lbs': 'sum',
        'compost_efficiency': 'mean'
    })
    
    write_to_report(report_file, "\nYearly Growth Analysis:")
    for col in ['lbs_collected', 'compost_created_lbs']:
        yearly_growth = year_stats[col].pct_change() * 100
        write_to_report(report_file, f"\n{col} Yearly Growth Rates:")
        write_to_report(report_file, yearly_growth.to_string())
    
    # Plot time series
    plt.figure(figsize=(15, 7))
    plt.plot(monthly_series.index, monthly_series['lbs_collected'], label='Waste Collected')
    plt.plot(monthly_series.index, monthly_series['compost_created_lbs'], label='Compost Created')
    plt.title('Waste Collection and Compost Production Over Time')
    plt.xlabel('Date')
    plt.ylabel('Pounds')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'time_series_trends.png'))
    plt.close()
    
    # Plot efficiency trend
    plt.figure(figsize=(15, 7))
    plt.plot(monthly_series.index, monthly_series['compost_efficiency'])
    plt.title('Compost Efficiency Over Time')
    plt.xlabel('Date')
    plt.ylabel('Efficiency')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'efficiency_trend.png'))
    plt.close()

if __name__ == "__main__":
    # Check if cleaned data exists, if not create it
    if not os.path.exists(os.path.join(OUTPUT_DIR, 'food_waste_clean.csv')):
        print("Cleaned data not found. Running data preparation first...")
        df = load_and_clean_data()
        save_processed_data(df)
    else:
        print("Loading cleaned data...")
        df = pd.read_csv(os.path.join(OUTPUT_DIR, 'food_waste_clean.csv'), parse_dates=['date_collection_datetime'])
    
    # Open report file
    with open(os.path.join(OUTPUT_DIR, 'descriptive_analysis_report.txt'), 'w') as f:
        # Write header
        write_to_report(f, "Food Waste Analysis Summary Report", "Executive Summary")
        write_to_report(f, f"Total Waste Collected: {df['lbs_collected'].sum():,.2f} lbs")
        write_to_report(f, f"Total Compost Created: {df['compost_created_lbs'].sum():,.2f} lbs")
        write_to_report(f, f"Average Efficiency: {df['compost_efficiency'].mean():.2%}")
        write_to_report(f, f"Number of Collection Sites: {df['stop_name'].nunique()}")
        write_to_report(f, f"Date Range: {df['date_collection_datetime'].min()} to {df['date_collection_datetime'].max()}")
        
        # Run and save detailed analyses
        print("Running descriptive analyses...")
        analyze_collection_volume(df, f)
        analyze_compost_efficiency(df, f)
        analyze_time_trends(df, f)
        
        print("Analysis complete! Check DESC_ANAL_OUTPUT directory for full results and generated plots.")
