#!/usr/bin/env python3
"""
NASA Lithium-Ion Battery Dataset Analysis and Preprocessing
B0005, B0006, B0018 batarya veri setleri için veri inceleme ve hazırlama
"""

import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_mat_file(filepath):
    """MAT dosyasını yükle ve veri yapısını döndür"""
    print(f"\n=== Loading {filepath.name} ===")
    try:
        mat_data = scipy.io.loadmat(filepath)
        
        # MATLAB metadata'sını kaldır
        data_keys = [key for key in mat_data.keys() if not key.startswith('__')]
        print(f"Available keys: {data_keys}")
        
        if data_keys:
            main_key = data_keys[0]  # Usually battery name like 'B0005'
            battery_data = mat_data[main_key]
            print(f"Main data structure: {main_key}")
            print(f"Shape: {battery_data.shape}")
            print(f"Data type: {battery_data.dtype}")
            
            return battery_data, main_key
        else:
            print("No data keys found!")
            return None, None
            
    except Exception as e:
        print(f"Error loading file: {e}")
        return None, None

def explore_battery_structure(battery_data, battery_name):
    """Batarya verisinin iç yapısını keşfet"""
    print(f"\n--- Exploring {battery_name} Structure ---")
    
    if battery_data.dtype.names:
        print(f"Top-level fields: {battery_data.dtype.names}")
        
        # Döngü verilerine erişim
        if 'cycle' in battery_data.dtype.names:
            cycle_data = battery_data['cycle'][0, 0]
            print(f"Cycle data shape: {cycle_data.shape}")
            
            # Number of cycles
            num_cycles = cycle_data.shape[1] if len(cycle_data.shape) > 1 else cycle_data.shape[0]
            print(f"Total cycles: {num_cycles}")
            
            # Examine first cycle
            if num_cycles > 0:
                first_cycle = cycle_data[0, 0] if len(cycle_data.shape) > 1 else cycle_data[0]
                if hasattr(first_cycle, 'dtype') and first_cycle.dtype.names:
                    print(f"Cycle fields: {first_cycle.dtype.names}")
                    
                    # Check cycle type
                    if 'type' in first_cycle.dtype.names:
                        cycle_type = first_cycle['type']
                        print(f"First cycle type: {cycle_type}")
                    
                    # Check data structure
                    if 'data' in first_cycle.dtype.names and len(first_cycle['data']) > 0:
                        data_struct = first_cycle['data'][0, 0]
                        if hasattr(data_struct, 'dtype') and data_struct.dtype.names:
                            print(f"Data fields: {list(data_struct.dtype.names)}")
            
            return cycle_data
    
    return None

def analyze_cycle_types(cycle_data, battery_name):
    """Analyze different types of cycles in the dataset"""
    print(f"\n--- Analyzing Cycle Types for {battery_name} ---")
    
    cycle_counts = {'charge': 0, 'discharge': 0, 'impedance': 0, 'other': 0}
    
    num_cycles = cycle_data.shape[1] if len(cycle_data.shape) > 1 else cycle_data.shape[0]
    
    for i in range(min(num_cycles, 1000)):  # Limit to first 1000 cycles for analysis
        try:
            cycle = cycle_data[0, i] if len(cycle_data.shape) > 1 else cycle_data[i]
            
            if hasattr(cycle, 'dtype') and 'type' in cycle.dtype.names:
                cycle_type_raw = cycle['type']
                
                # Extract string from numpy array structure
                if hasattr(cycle_type_raw, 'shape') and len(cycle_type_raw) > 0:
                    cycle_type = str(cycle_type_raw[0][0]) if cycle_type_raw[0].size > 0 else 'unknown'
                else:
                    cycle_type = str(cycle_type_raw)
                
                # Count cycle types
                if 'charge' in cycle_type.lower():
                    cycle_counts['charge'] += 1
                elif 'discharge' in cycle_type.lower():
                    cycle_counts['discharge'] += 1
                elif 'impedance' in cycle_type.lower():
                    cycle_counts['impedance'] += 1
                else:
                    cycle_counts['other'] += 1
                    
        except Exception as e:
            print(f"Error processing cycle {i}: {e}")
            continue
    
    print("Cycle type distribution:")
    for cycle_type, count in cycle_counts.items():
        if count > 0:
            print(f"  {cycle_type}: {count} cycles")
    
    return cycle_counts

def extract_discharge_features(cycle_data, battery_name):
    """Extract features from discharge cycles for SoH/SoC analysis"""
    print(f"\n--- Extracting Discharge Features for {battery_name} ---")
    
    discharge_features = []
    num_cycles = cycle_data.shape[1] if len(cycle_data.shape) > 1 else cycle_data.shape[0]
    
    discharge_count = 0
    
    for i in range(num_cycles):
        try:
            cycle = cycle_data[0, i] if len(cycle_data.shape) > 1 else cycle_data[i]
            
            # Check if this is a discharge cycle
            if hasattr(cycle, 'dtype') and 'type' in cycle.dtype.names:
                cycle_type_raw = cycle['type']
                cycle_type = str(cycle_type_raw[0][0]) if (hasattr(cycle_type_raw, 'shape') 
                                                         and len(cycle_type_raw) > 0 
                                                         and cycle_type_raw[0].size > 0) else 'unknown'
                
                if 'discharge' not in cycle_type.lower():
                    continue
                
                discharge_count += 1
                
                # Extract ambient temperature
                ambient_temp = np.nan
                if 'ambient_temperature' in cycle.dtype.names:
                    temp_raw = cycle['ambient_temperature']
                    if hasattr(temp_raw, 'shape') and len(temp_raw) > 0:
                        ambient_temp = float(temp_raw[0]) if temp_raw[0].size > 0 else np.nan
                
                # Extract measurement data
                if 'data' in cycle.dtype.names and len(cycle['data']) > 0:
                    data_struct = cycle['data'][0, 0]
                    
                    if hasattr(data_struct, 'dtype') and data_struct.dtype.names:
                        # Extract key measurements
                        features = {
                            'battery': battery_name,
                            'cycle_number': i,
                            'ambient_temperature': ambient_temp,
                        }
                        
                        # Voltage measurements
                        if 'Voltage_measured' in data_struct.dtype.names:
                            voltage = data_struct['Voltage_measured'].flatten()
                            features.update({
                                'voltage_start': voltage[0] if len(voltage) > 0 else np.nan,
                                'voltage_end': voltage[-1] if len(voltage) > 0 else np.nan,
                                'voltage_min': np.min(voltage) if len(voltage) > 0 else np.nan,
                                'voltage_mean': np.mean(voltage) if len(voltage) > 0 else np.nan,
                                'voltage_std': np.std(voltage) if len(voltage) > 0 else np.nan,
                            })
                        
                        # Current measurements
                        if 'Current_measured' in data_struct.dtype.names:
                            current = data_struct['Current_measured'].flatten()
                            features.update({
                                'current_mean': np.mean(current) if len(current) > 0 else np.nan,
                                'current_std': np.std(current) if len(current) > 0 else np.nan,
                            })
                        
                        # Temperature measurements
                        if 'Temperature_measured' in data_struct.dtype.names:
                            temperature = data_struct['Temperature_measured'].flatten()
                            features.update({
                                'temperature_start': temperature[0] if len(temperature) > 0 else np.nan,
                                'temperature_end': temperature[-1] if len(temperature) > 0 else np.nan,
                                'temperature_mean': np.mean(temperature) if len(temperature) > 0 else np.nan,
                                'temperature_max': np.max(temperature) if len(temperature) > 0 else np.nan,
                            })
                        
                        # Time measurements
                        if 'Time' in data_struct.dtype.names:
                            time = data_struct['Time'].flatten()
                            features['discharge_duration'] = time[-1] - time[0] if len(time) > 1 else 0
                        
                        # Capacity (key for SoH analysis)
                        if 'Capacity' in data_struct.dtype.names:
                            capacity = data_struct['Capacity'].flatten()
                            features['capacity_delivered'] = capacity[-1] if len(capacity) > 0 else np.nan
                        
                        discharge_features.append(features)
                        
                        if discharge_count % 50 == 0:
                            print(f"  Processed {discharge_count} discharge cycles...")
        
        except Exception as e:
            print(f"Error processing cycle {i}: {e}")
            continue
    
    print(f"Extracted features from {len(discharge_features)} discharge cycles")
    return discharge_features

def calculate_soh_indicators(discharge_features):
    """Calculate State of Health indicators"""
    print("\n--- Calculating SoH Indicators ---")
    
    if not discharge_features:
        return []
    
    # Group by battery
    battery_groups = {}
    for feature in discharge_features:
        battery = feature['battery']
        if battery not in battery_groups:
            battery_groups[battery] = []
        battery_groups[battery].append(feature)
    
    # Calculate SoH for each battery
    for battery_name, battery_data in battery_groups.items():
        # Sort by cycle number
        battery_data.sort(key=lambda x: x['cycle_number'])
        
        # Calculate initial capacity (average of first few valid cycles)
        valid_capacities = [d['capacity_delivered'] for d in battery_data[:10] 
                          if not np.isnan(d.get('capacity_delivered', np.nan))]
        
        if valid_capacities:
            initial_capacity = np.mean(valid_capacities)
            print(f"{battery_name} initial capacity: {initial_capacity:.3f} Ah")
            
            # Calculate SoH for each cycle
            for data in battery_data:
                if not np.isnan(data.get('capacity_delivered', np.nan)):
                    # SoH = current capacity / initial capacity * 100
                    data['soh'] = (data['capacity_delivered'] / initial_capacity) * 100
                    # Capacity fade = (initial - current) / initial * 100
                    data['capacity_fade'] = ((initial_capacity - data['capacity_delivered']) / initial_capacity) * 100
                else:
                    data['soh'] = np.nan
                    data['capacity_fade'] = np.nan
    
    # Flatten the results
    result = []
    for battery_data in battery_groups.values():
        result.extend(battery_data)
    
    return result

def create_summary_dataframe(all_features):
    """Create a clean pandas DataFrame from extracted features"""
    print("\n--- Creating Summary DataFrame ---")
    
    if not all_features:
        print("No features to process!")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_features)
    
    print(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    
    # Data quality check
    print("\nData Quality Summary:")
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            non_null = df[col].notna().sum()
            print(f"  {col}: {non_null}/{len(df)} non-null values ({non_null/len(df)*100:.1f}%)")
    
    return df

def clean_and_preprocess_data(df):
    """Clean and preprocess the data"""
    print("\n--- Data Cleaning and Preprocessing ---")
    
    if df.empty:
        return df
    
    original_count = len(df)
    
    # Remove rows with critical missing values
    critical_columns = ['voltage_start', 'voltage_end']
    df_clean = df.dropna(subset=critical_columns)
    
    removed_count = original_count - len(df_clean)
    if removed_count > 0:
        print(f"Removed {removed_count} rows with missing critical values")
    
    # Handle capacity outliers
    if 'capacity_delivered' in df_clean.columns:
        capacity_col = df_clean['capacity_delivered']
        Q1 = capacity_col.quantile(0.25)
        Q3 = capacity_col.quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = ((capacity_col < lower_bound) | (capacity_col > upper_bound))
        print(f"Found {outliers.sum()} capacity outliers (keeping them flagged)")
        
        df_clean['is_capacity_outlier'] = outliers
    
    # Fill missing ambient temperature with median
    if 'ambient_temperature' in df_clean.columns:
        median_temp = df_clean['ambient_temperature'].median()
        df_clean['ambient_temperature'] = df_clean['ambient_temperature'].fillna(median_temp)
        print(f"Filled missing ambient temperatures with median: {median_temp:.1f}°C")
    
    print(f"Final clean dataset: {len(df_clean)} rows")
    return df_clean

def create_visualizations(df):
    """Create exploratory visualizations"""
    print("\n--- Creating Visualizations ---")
    
    if df.empty:
        print("No data to visualize!")
        return
    
    # Set up matplotlib
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('NASA Battery Dataset - Exploratory Analysis', fontsize=14)
    
    # 1. Capacity delivered over cycles
    if 'capacity_delivered' in df.columns:
        for battery in df['battery'].unique():
            battery_data = df[df['battery'] == battery].sort_values('cycle_number')
            axes[0, 0].plot(battery_data['cycle_number'], battery_data['capacity_delivered'], 
                           'o-', label=battery, markersize=3, alpha=0.7)
        
        axes[0, 0].set_title('Capacity Delivered Over Cycles')
        axes[0, 0].set_xlabel('Cycle Number')
        axes[0, 0].set_ylabel('Capacity (Ah)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. State of Health
    if 'soh' in df.columns:
        for battery in df['battery'].unique():
            battery_data = df[df['battery'] == battery].sort_values('cycle_number')
            battery_data_clean = battery_data.dropna(subset=['soh'])
            if not battery_data_clean.empty:
                axes[0, 1].plot(battery_data_clean['cycle_number'], battery_data_clean['soh'], 
                               'o-', label=battery, markersize=3, alpha=0.7)
        
        axes[0, 1].set_title('State of Health (SoH) Over Cycles')
        axes[0, 1].set_xlabel('Cycle Number')
        axes[0, 1].set_ylabel('SoH (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=70, color='red', linestyle='--', alpha=0.7, label='EOL Threshold')
    
    # 3. Voltage characteristics
    if 'voltage_start' in df.columns and 'voltage_end' in df.columns:
        scatter = axes[1, 0].scatter(df['voltage_start'], df['voltage_end'], 
                                   c=df['cycle_number'], cmap='viridis', alpha=0.6, s=20)
        axes[1, 0].set_title('Discharge Voltage: Start vs End')
        axes[1, 0].set_xlabel('Start Voltage (V)')
        axes[1, 0].set_ylabel('End Voltage (V)')
        axes[1, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1, 0], label='Cycle Number')
    
    # 4. Temperature analysis
    if 'temperature_mean' in df.columns:
        df.boxplot(column='temperature_mean', by='battery', ax=axes[1, 1])
        axes[1, 1].set_title('Average Temperature by Battery')
        axes[1, 1].set_ylabel('Temperature (°C)')
    
    plt.tight_layout()
    plt.savefig('nasa_battery_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualization saved as 'nasa_battery_analysis.png'")

def main():
    """Main analysis function"""
    print("NASA Lithium-Ion Battery Dataset Analysis")
    print("="*60)
    
    # Define data files
    data_dir = Path("data")
    battery_files = ["B0005.mat", "B0006.mat", "B0018.mat"]
    
    all_discharge_features = []
    
    # Process each battery file
    for battery_file in battery_files:
        file_path = data_dir / battery_file
        
        if not file_path.exists():
            print(f"WARNING: {file_path} not found!")
            continue
        
        # Load and explore data structure
        battery_data, battery_name = load_mat_file(file_path)
        
        if battery_data is not None:
            # Explore structure
            cycle_data = explore_battery_structure(battery_data, battery_name)
            
            if cycle_data is not None:
                # Analyze cycle types
                cycle_counts = analyze_cycle_types(cycle_data, battery_name)
                
                # Extract discharge features
                discharge_features = extract_discharge_features(cycle_data, battery_name)
                all_discharge_features.extend(discharge_features)
    
    # Calculate SoH indicators
    all_discharge_features = calculate_soh_indicators(all_discharge_features)
    
    # Create DataFrame
    df = create_summary_dataframe(all_discharge_features)
    
    if not df.empty:
        # Clean and preprocess
        df_clean = clean_and_preprocess_data(df)
        
        # Create visualizations
        create_visualizations(df_clean)
        
        # Save processed data
        df_clean.to_csv('processed_battery_data.csv', index=False)
        print(f"\nSaved processed data to 'processed_battery_data.csv'")
        
        # Final summary
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        
        for battery in df_clean['battery'].unique():
            battery_data = df_clean[df_clean['battery'] == battery]
            print(f"\n{battery}:")
            print(f"  Discharge cycles: {len(battery_data)}")
            
            if 'capacity_delivered' in battery_data.columns:
                cap_stats = battery_data['capacity_delivered'].describe()
                print(f"  Capacity - Min: {cap_stats['min']:.3f} Ah, Max: {cap_stats['max']:.3f} Ah")
            
            if 'soh' in battery_data.columns:
                soh_valid = battery_data['soh'].dropna()
                if not soh_valid.empty:
                    print(f"  SoH - Min: {soh_valid.min():.1f}%, Max: {soh_valid.max():.1f}%")
        
        return df_clean
    
    else:
        print("No data processed successfully!")
        return None

if __name__ == "__main__":
    processed_data = main()
