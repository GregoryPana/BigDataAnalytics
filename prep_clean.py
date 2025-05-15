import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler

def load_and_clean_data(file_path='food-waste-raw.csv'):
    """
    Load and clean the food waste data following CRISP-DM methodology
    """
    # Load the data
    print("Loading data...")
    df = pd.read_csv(file_path)
    
    # Data Quality Report
    initial_rows = len(df)
    print(f"\nInitial Data Quality Report:")
    print(f"Total records: {initial_rows}")
    print(f"Missing values:\n{df.isnull().sum()}")
    
    # Remove rows with missing values
    print("\nRemoving missing values...")
    df = df.dropna()
    print(f"Rows removed due to missing values: {initial_rows - len(df)}")
    
    # Convert date columns to datetime
    print("\nConverting date columns...")
    df['date_collection_datetime'] = pd.to_datetime(df['date_collection_datetime'], 
                                                  dayfirst=True, 
                                                  errors='coerce')
    
    # Remove rows with invalid dates
    invalid_dates = df['date_collection_datetime'].isnull()
    df = df[~invalid_dates]
    print(f"Rows removed due to invalid dates: {sum(invalid_dates)}")
    
    # Remove duplicates
    print("\nRemoving duplicates...")
    duplicates = df.duplicated()
    df = df.drop_duplicates()
    print(f"Duplicate rows removed: {sum(duplicates)}")
    
    # Validate and clean numeric data
    print("\nValidating numeric data...")
    # Remove negative or zero values
    invalid_collected = (df['lbs_collected'] <= 0)
    invalid_compost = (df['compost_created_lbs'] <= 0)
    df = df[~(invalid_collected | invalid_compost)]
    print(f"Rows removed due to invalid weights: {sum(invalid_collected | invalid_compost)}")
    
    # Remove outliers (values more than 3 std devs from mean)
    for col in ['lbs_collected', 'compost_created_lbs']:
        mean = df[col].mean()
        std = df[col].std()
        outliers = (df[col] < mean - 3*std) | (df[col] > mean + 3*std)
        df = df[~outliers]
        print(f"Outliers removed from {col}: {sum(outliers)}")
    
    # Calculate efficiency metrics
    print("\nCalculating efficiency metrics...")
    df['compost_efficiency'] = df['compost_created_lbs'] / df['lbs_collected']
    
    # Remove unrealistic efficiency values
    invalid_efficiency = (df['compost_efficiency'] <= 0) | (df['compost_efficiency'] > 1)
    df = df[~invalid_efficiency]
    print(f"Rows removed due to invalid efficiency: {sum(invalid_efficiency)}")
    
    # Add temporal features
    print("\nAdding temporal features...")
    df['year'] = df['date_collection_datetime'].dt.year
    df['month'] = df['date_collection_datetime'].dt.month
    df['day_of_week'] = df['date_collection_datetime'].dt.dayofweek
    
    # Normalize numeric features
    print("\nNormalizing features...")
    scaler = StandardScaler()
    df[['lbs_collected_normalized', 'compost_created_lbs_normalized']] = scaler.fit_transform(
        df[['lbs_collected', 'compost_created_lbs']])
    
    # Final data quality report
    print(f"\nFinal Data Quality Report:")
    print(f"Initial records: {initial_rows}")
    print(f"Final records: {len(df)}")
    print(f"Records removed: {initial_rows - len(df)}")
    print(f"Remaining data by year:\n{df['year'].value_counts().sort_index()}")
    print("\nData preparation completed!")
    
    return df

def save_processed_data(df, output_file='food_waste_clean.csv'):
    """
    Save the processed data to a CSV file
    """
    df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")

if __name__ == "__main__":
    # Execute data preparation pipeline
    processed_df = load_and_clean_data()
    save_processed_data(processed_df)
    
    # Print data quality report
    print("\nData Quality Report:")
    print("-" * 50)
    print(f"Total number of records: {len(processed_df)}")
    print(f"Number of features: {processed_df.shape[1]}")
    print("\nMissing values:")
    print(processed_df.isnull().sum())
    print("\nData types:")
    print(processed_df.dtypes)
