#!/usr/bin/env python3
"""
NASA Lithium-Ion Battery Dataset - Complete Preprocessing
Corrected version with proper cycle type detection
"""

import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_and_analyze_battery(file_path):
    """Batarya verisini yükle ve tüm döngü bilgilerini çıkar"""
    print(f"\n=== Processing {file_path.name} ===")
    
    try:
        # MATLAB dosyasını yükle
        mat_data = scipy.io.loadmat(file_path)
        battery_name = file_path.stem
        battery_data = mat_data[battery_name]
        cycle_data = battery_data['cycle'][0, 0]
        
        print(f"Battery: {battery_name}")
        print(f"Total cycles: {cycle_data.shape[1]}")
        
        return process_all_cycles(cycle_data, battery_name)
        
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

def process_all_cycles(cycle_data, battery_name):
    """Process all cycles and extract features"""
    
    discharge_cycles = []
    charge_cycles = []
    impedance_cycles = []
    
    num_cycles = cycle_data.shape[1]
    print(f"Processing {num_cycles} cycles...")
    
    for i in range(num_cycles):
        try:
            cycle = cycle_data[0, i]
            
            # Extract cycle type
            cycle_type = str(cycle['type'][0])  # Fixed: direct access to string
            
            # Extract ambient temperature
            ambient_temp = float(cycle['ambient_temperature'][0]) if cycle['ambient_temperature'].size > 0 else np.nan
            
            # Extract cycle time
            cycle_time = cycle['time'] if cycle['time'].size > 0 else None
            
            # Process based on cycle type
            if cycle_type == 'discharge':
                features = extract_discharge_features(cycle, i, battery_name, ambient_temp)
                if features:
                    discharge_cycles.append(features)
                    
            elif cycle_type == 'charge':
                features = extract_charge_features(cycle, i, battery_name, ambient_temp)
                if features:
                    charge_cycles.append(features)
                    
            elif cycle_type == 'impedance':
                features = extract_impedance_features(cycle, i, battery_name, ambient_temp)
                if features:
                    impedance_cycles.append(features)
            
        except Exception as e:
            print(f"Error processing cycle {i}: {e}")
            continue
    
    print(f"Extracted:")
    print(f"  - {len(discharge_cycles)} discharge cycles")
    print(f"  - {len(charge_cycles)} charge cycles") 
    print(f"  - {len(impedance_cycles)} impedance cycles")
    
    return discharge_cycles, charge_cycles, impedance_cycles

def extract_discharge_features(cycle, cycle_num, battery_name, ambient_temp):
    """Extract features from discharge cycle"""
    
    if 'data' not in cycle.dtype.names or len(cycle['data']) == 0:
        return None
    
    data = cycle['data'][0, 0]
    
    # Extract time series data
    time = data['Time'].flatten()
    voltage = data['Voltage_measured'].flatten()
    current = data['Current_measured'].flatten()
    temperature = data['Temperature_measured'].flatten()
    
    # Capacity is available in discharge cycles
    capacity = data['Capacity'].flatten() if 'Capacity' in data.dtype.names else None
    
    # Calculate features
    features = {
        'battery': battery_name,
        'cycle_number': cycle_num,
        'cycle_type': 'discharge',
        'ambient_temperature': ambient_temp,
        
        # Time-based features
        'discharge_duration': float(time[-1] - time[0]) if len(time) > 1 else 0,
        'data_points': len(time),
        
        # Voltage features
        'voltage_start': float(voltage[0]),
        'voltage_end': float(voltage[-1]),
        'voltage_min': float(np.min(voltage)),
        'voltage_max': float(np.max(voltage)),
        'voltage_mean': float(np.mean(voltage)),
        'voltage_std': float(np.std(voltage)),
        
        # Current features (discharge current is typically negative)
        'current_mean': float(np.mean(current)),
        'current_std': float(np.std(current)),
        'current_min': float(np.min(current)),
        'current_max': float(np.max(current)),
        
        # Temperature features
        'temp_start': float(temperature[0]),
        'temp_end': float(temperature[-1]),
        'temp_mean': float(np.mean(temperature)),
        'temp_max': float(np.max(temperature)),
        'temp_rise': float(temperature[-1] - temperature[0]),
        
        # Energy calculation (V*I integrated over time)
        'energy_delivered': calculate_energy(voltage, current, time),
        
        # Capacity features (key for SoH)
        'capacity_delivered': float(capacity[-1]) if capacity is not None else np.nan,
        'capacity_delivered_cumulative': float(capacity[-1]) if capacity is not None else np.nan,
    }
    
    return features

def extract_charge_features(cycle, cycle_num, battery_name, ambient_temp):
    """Extract features from charge cycle"""
    
    if 'data' not in cycle.dtype.names or len(cycle['data']) == 0:
        return None
    
    data = cycle['data'][0, 0]
    
    # Extract time series data
    time = data['Time'].flatten()
    voltage = data['Voltage_measured'].flatten()
    current = data['Current_measured'].flatten()
    temperature = data['Temperature_measured'].flatten()
    
    features = {
        'battery': battery_name,
        'cycle_number': cycle_num,
        'cycle_type': 'charge',
        'ambient_temperature': ambient_temp,
        
        # Charge duration
        'charge_duration': float(time[-1] - time[0]) if len(time) > 1 else 0,
        
        # Voltage features
        'voltage_start': float(voltage[0]),
        'voltage_end': float(voltage[-1]),
        'voltage_max': float(np.max(voltage)),
        
        # Current features (charge current is typically positive)
        'current_start': float(current[0]),
        'current_end': float(current[-1]),
        'current_mean': float(np.mean(current)),
        
        # Energy charged
        'energy_charged': calculate_energy(voltage, current, time),
        
        # Temperature
        'temp_start': float(temperature[0]),
        'temp_end': float(temperature[-1]),
        'temp_rise': float(temperature[-1] - temperature[0]),
    }
    
    return features

def extract_impedance_features(cycle, cycle_num, battery_name, ambient_temp):
    """Extract features from impedance cycle"""
    
    if 'data' not in cycle.dtype.names or len(cycle['data']) == 0:
        return None
    
    data = cycle['data'][0, 0]
    
    # Impedance cycles have different data fields
    features = {
        'battery': battery_name,
        'cycle_number': cycle_num,
        'cycle_type': 'impedance',
        'ambient_temperature': ambient_temp,
    }
    
    # Extract impedance-specific measurements
    if 'Battery_impedance' in data.dtype.names:
        impedance = data['Battery_impedance'].flatten()
        features.update({
            'battery_impedance_mean': float(np.mean(impedance)),
            'battery_impedance_std': float(np.std(impedance)),
        })
    
    if 'Rectified_Impedance' in data.dtype.names:
        rect_impedance = data['Rectified_Impedance'].flatten()
        features.update({
            'rectified_impedance_mean': float(np.mean(rect_impedance)),
        })
    
    # Electrolyte and charge transfer resistance
    if 'Re' in data.dtype.names:
        re = data['Re'].flatten()
        features['electrolyte_resistance'] = float(np.mean(re))
    
    if 'Rct' in data.dtype.names:
        rct = data['Rct'].flatten()
        features['charge_transfer_resistance'] = float(np.mean(rct))
    
    return features

def calculate_energy(voltage, current, time):
    """Calculate energy from voltage, current and time arrays"""
    try:
        if len(voltage) > 1 and len(current) > 1 and len(time) > 1:
            # Energy = integral of V*I dt (in Wh)
            power = voltage * np.abs(current)
            energy_wh = np.trapz(power, time) / 3600  # Convert from Ws to Wh
            return float(energy_wh)
    except:
        pass
    return np.nan

def calculate_soh_features(discharge_cycles):
    """Calculate State of Health features for discharge cycles"""
    print("\n--- Calculating SoH Features ---")
    
    # Group by battery
    battery_groups = {}
    for cycle in discharge_cycles:
        battery = cycle['battery']
        if battery not in battery_groups:
            battery_groups[battery] = []
        battery_groups[battery].append(cycle)
    
    # Calculate SoH for each battery
    for battery_name, battery_cycles in battery_groups.items():
        # Sort by cycle number
        battery_cycles.sort(key=lambda x: x['cycle_number'])
        
        # Calculate initial capacity (average of first few valid cycles)
        valid_capacities = []
        for cycle in battery_cycles[:5]:
            if not np.isnan(cycle['capacity_delivered']):
                valid_capacities.append(cycle['capacity_delivered'])
        
        if valid_capacities:
            initial_capacity = np.mean(valid_capacities)
            print(f"{battery_name} initial capacity: {initial_capacity:.3f} Ah")
            
            # Calculate SoH for each cycle
            for cycle in battery_cycles:
                if not np.isnan(cycle['capacity_delivered']):
                    # SoH = (current capacity / initial capacity) * 100
                    cycle['soh'] = (cycle['capacity_delivered'] / initial_capacity) * 100
                    
                    # Capacity fade percentage
                    cycle['capacity_fade'] = ((initial_capacity - cycle['capacity_delivered']) / initial_capacity) * 100
                    
                    # Relative capacity (normalized to initial)
                    cycle['relative_capacity'] = cycle['capacity_delivered'] / initial_capacity
                else:
                    cycle['soh'] = np.nan
                    cycle['capacity_fade'] = np.nan
                    cycle['relative_capacity'] = np.nan
    
    return discharge_cycles

def create_comprehensive_dataset(all_discharge_cycles, all_charge_cycles, all_impedance_cycles):
    """Create comprehensive dataset with all cycle types"""
    
    print("\n--- Creating Comprehensive Dataset ---")
    
    # Convert to DataFrames
    discharge_df = pd.DataFrame(all_discharge_cycles) if all_discharge_cycles else pd.DataFrame()
    charge_df = pd.DataFrame(all_charge_cycles) if all_charge_cycles else pd.DataFrame()
    impedance_df = pd.DataFrame(all_impedance_cycles) if all_impedance_cycles else pd.DataFrame()
    
    print(f"Dataset sizes:")
    print(f"  Discharge: {len(discharge_df)} cycles")
    print(f"  Charge: {len(charge_df)} cycles") 
    print(f"  Impedance: {len(impedance_df)} cycles")
    
    return discharge_df, charge_df, impedance_df

def data_quality_analysis(discharge_df):
    """Perform data quality analysis"""
    
    print("\n--- Data Quality Analysis ---")
    
    if discharge_df.empty:
        print("No discharge data available!")
        return discharge_df
    
    print(f"Total discharge cycles: {len(discharge_df)}")
    
    # Check missing values
    missing_summary = discharge_df.isnull().sum()
    critical_missing = missing_summary[missing_summary > 0]
    
    if len(critical_missing) > 0:
        print("Missing values:")
        for col, count in critical_missing.items():
            print(f"  {col}: {count} ({count/len(discharge_df)*100:.1f}%)")
    
    # Capacity analysis
    if 'capacity_delivered' in discharge_df.columns:
        capacity_stats = discharge_df['capacity_delivered'].describe()
        print(f"\nCapacity statistics:")
        print(f"  Mean: {capacity_stats['mean']:.3f} Ah")
        print(f"  Std: {capacity_stats['std']:.3f} Ah") 
        print(f"  Range: {capacity_stats['min']:.3f} - {capacity_stats['max']:.3f} Ah")
        
        # Detect outliers
        Q1 = discharge_df['capacity_delivered'].quantile(0.25)
        Q3 = discharge_df['capacity_delivered'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = ((discharge_df['capacity_delivered'] < lower_bound) | 
                   (discharge_df['capacity_delivered'] > upper_bound))
        print(f"  Capacity outliers: {outliers.sum()} ({outliers.sum()/len(discharge_df)*100:.1f}%)")
        
        discharge_df['is_capacity_outlier'] = outliers
    
    return discharge_df

def create_advanced_visualizations(discharge_df, charge_df, impedance_df):
    """Create comprehensive visualizations"""
    
    print("\n--- Creating Advanced Visualizations ---")
    
    plt.style.use('default')
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle('NASA Battery Dataset - Comprehensive Analysis', fontsize=16)
    
    # 1. Capacity degradation over cycles
    if not discharge_df.empty:
        for battery in discharge_df['battery'].unique():
            battery_data = discharge_df[discharge_df['battery'] == battery].sort_values('cycle_number')
            axes[0, 0].plot(battery_data['cycle_number'], battery_data['capacity_delivered'], 
                           'o-', label=battery, markersize=2, alpha=0.8)
        
        axes[0, 0].set_title('Capacity Degradation')
        axes[0, 0].set_xlabel('Cycle Number')
        axes[0, 0].set_ylabel('Capacity (Ah)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. State of Health
    if not discharge_df.empty and 'soh' in discharge_df.columns:
        for battery in discharge_df['battery'].unique():
            battery_data = discharge_df[discharge_df['battery'] == battery].sort_values('cycle_number')
            valid_data = battery_data.dropna(subset=['soh'])
            if not valid_data.empty:
                axes[0, 1].plot(valid_data['cycle_number'], valid_data['soh'], 
                               'o-', label=battery, markersize=2, alpha=0.8)
        
        axes[0, 1].set_title('State of Health (SoH)')
        axes[0, 1].set_xlabel('Cycle Number')
        axes[0, 1].set_ylabel('SoH (%)')
        axes[0, 1].axhline(y=70, color='red', linestyle='--', alpha=0.7, label='EOL (70%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Voltage characteristics
    if not discharge_df.empty:
        scatter = axes[0, 2].scatter(discharge_df['voltage_start'], discharge_df['voltage_end'], 
                                   c=discharge_df['cycle_number'], cmap='viridis', alpha=0.6, s=10)
        axes[0, 2].set_title('Discharge Voltage Profile')
        axes[0, 2].set_xlabel('Start Voltage (V)')
        axes[0, 2].set_ylabel('End Voltage (V)')
        axes[0, 2].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0, 2], label='Cycle')
    
    # 4. Temperature analysis
    if not discharge_df.empty:
        for battery in discharge_df['battery'].unique():
            battery_data = discharge_df[discharge_df['battery'] == battery].sort_values('cycle_number')
            axes[1, 0].plot(battery_data['cycle_number'], battery_data['temp_mean'], 
                           'o-', label=battery, markersize=2, alpha=0.8)
        
        axes[1, 0].set_title('Temperature Evolution')
        axes[1, 0].set_xlabel('Cycle Number')
        axes[1, 0].set_ylabel('Average Temperature (°C)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Energy delivered
    if not discharge_df.empty and 'energy_delivered' in discharge_df.columns:
        valid_energy = discharge_df.dropna(subset=['energy_delivered'])
        if not valid_energy.empty:
            for battery in valid_energy['battery'].unique():
                battery_data = valid_energy[valid_energy['battery'] == battery].sort_values('cycle_number')
                axes[1, 1].plot(battery_data['cycle_number'], battery_data['energy_delivered'], 
                               'o-', label=battery, markersize=2, alpha=0.8)
        
        axes[1, 1].set_title('Energy Delivered per Cycle')
        axes[1, 1].set_xlabel('Cycle Number')
        axes[1, 1].set_ylabel('Energy (Wh)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Discharge duration
    if not discharge_df.empty:
        discharge_df.boxplot(column='discharge_duration', by='battery', ax=axes[1, 2])
        axes[1, 2].set_title('Discharge Duration by Battery')
        axes[1, 2].set_ylabel('Duration (seconds)')
    
    # 7. Impedance analysis (if available)
    if not impedance_df.empty and 'battery_impedance_mean' in impedance_df.columns:
        for battery in impedance_df['battery'].unique():
            battery_data = impedance_df[impedance_df['battery'] == battery].sort_values('cycle_number')
            axes[2, 0].plot(battery_data['cycle_number'], battery_data['battery_impedance_mean'], 
                           'o-', label=battery, markersize=2, alpha=0.8)
        
        axes[2, 0].set_title('Battery Impedance Evolution')
        axes[2, 0].set_xlabel('Cycle Number')
        axes[2, 0].set_ylabel('Impedance (Ohms)')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
    
    # 8. Charge-discharge comparison
    if not charge_df.empty and not discharge_df.empty:
        # Merge charge and discharge data by cycle number and battery
        charge_summary = charge_df.groupby(['battery', 'cycle_number'])['energy_charged'].mean().reset_index()
        discharge_summary = discharge_df.groupby(['battery', 'cycle_number'])['energy_delivered'].mean().reset_index()
        
        merged = pd.merge(charge_summary, discharge_summary, on=['battery', 'cycle_number'], how='inner')
        if not merged.empty:
            axes[2, 1].scatter(merged['energy_charged'], merged['energy_delivered'], 
                              alpha=0.6, s=10)
            axes[2, 1].plot([0, merged[['energy_charged', 'energy_delivered']].max().max()], 
                           [0, merged[['energy_charged', 'energy_delivered']].max().max()], 
                           'r--', alpha=0.7)
            axes[2, 1].set_title('Charge vs Discharge Energy')
            axes[2, 1].set_xlabel('Energy Charged (Wh)')
            axes[2, 1].set_ylabel('Energy Delivered (Wh)')
            axes[2, 1].grid(True, alpha=0.3)
    
    # 9. Capacity vs SoH correlation
    if not discharge_df.empty and 'soh' in discharge_df.columns:
        valid_data = discharge_df.dropna(subset=['capacity_delivered', 'soh'])
        if not valid_data.empty:
            axes[2, 2].scatter(valid_data['capacity_delivered'], valid_data['soh'], alpha=0.6, s=10)
            axes[2, 2].set_title('Capacity vs SoH')
            axes[2, 2].set_xlabel('Capacity Delivered (Ah)')
            axes[2, 2].set_ylabel('SoH (%)')
            axes[2, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('nasa_battery_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Comprehensive visualization saved as 'nasa_battery_comprehensive_analysis.png'")

def save_processed_datasets(discharge_df, charge_df, impedance_df):
    """Save all processed datasets"""
    
    print("\n--- Saving Processed Datasets ---")
    
    # Save as CSV files
    if not discharge_df.empty:
        discharge_df.to_csv('nasa_discharge_cycles.csv', index=False)
        print(f"Saved discharge data: {len(discharge_df)} cycles -> nasa_discharge_cycles.csv")
    
    if not charge_df.empty:
        charge_df.to_csv('nasa_charge_cycles.csv', index=False)
        print(f"Saved charge data: {len(charge_df)} cycles -> nasa_charge_cycles.csv")
    
    if not impedance_df.empty:
        impedance_df.to_csv('nasa_impedance_cycles.csv', index=False)
        print(f"Saved impedance data: {len(impedance_df)} cycles -> nasa_impedance_cycles.csv")
    
    # Create a combined summary
    summary_data = []
    
    for battery_file in ['B0005', 'B0006', 'B0018']:
        if not discharge_df.empty:
            battery_discharge = discharge_df[discharge_df['battery'] == battery_file]
            summary_data.append({
                'battery': battery_file,
                'total_discharge_cycles': len(battery_discharge),
                'initial_capacity': battery_discharge['capacity_delivered'].iloc[0] if not battery_discharge.empty else np.nan,
                'final_capacity': battery_discharge['capacity_delivered'].iloc[-1] if not battery_discharge.empty else np.nan,
                'capacity_fade_percent': battery_discharge['capacity_fade'].iloc[-1] if 'capacity_fade' in battery_discharge.columns and not battery_discharge.empty else np.nan,
                'min_soh': battery_discharge['soh'].min() if 'soh' in battery_discharge.columns and not battery_discharge.empty else np.nan,
                'cycles_to_eol': len(battery_discharge[battery_discharge['soh'] > 70]) if 'soh' in battery_discharge.columns and not battery_discharge.empty else np.nan
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('nasa_battery_summary.csv', index=False)
        print("Saved battery summary -> nasa_battery_summary.csv")

def main():
    """Main preprocessing and analysis function"""
    
    print("NASA Lithium-Ion Battery Dataset - Complete Analysis")
    print("="*70)
    
    # Define data files
    data_dir = Path("data")
    battery_files = ["B0005.mat", "B0006.mat", "B0018.mat"]
    
    all_discharge_cycles = []
    all_charge_cycles = []
    all_impedance_cycles = []
    
    # Process each battery file
    for battery_file in battery_files:
        file_path = data_dir / battery_file
        
        if not file_path.exists():
            print(f"WARNING: {file_path} not found!")
            continue
        
        # Load and process battery data
        discharge_cycles, charge_cycles, impedance_cycles = load_and_analyze_battery(file_path)
        
        all_discharge_cycles.extend(discharge_cycles)
        all_charge_cycles.extend(charge_cycles)
        all_impedance_cycles.extend(impedance_cycles)
    
    # Calculate SoH features
    all_discharge_cycles = calculate_soh_features(all_discharge_cycles)
    
    # Create datasets
    discharge_df, charge_df, impedance_df = create_comprehensive_dataset(
        all_discharge_cycles, all_charge_cycles, all_impedance_cycles)
    
    # Data quality analysis
    discharge_df = data_quality_analysis(discharge_df)
    
    # Create visualizations
    create_advanced_visualizations(discharge_df, charge_df, impedance_df)
    
    # Save datasets
    save_processed_datasets(discharge_df, charge_df, impedance_df)
    
    # Final summary
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE - SUMMARY")
    print("="*70)
    
    print(f"\nTotal processed cycles:")
    print(f"  Discharge: {len(discharge_df)}")
    print(f"  Charge: {len(charge_df)}")
    print(f"  Impedance: {len(impedance_df)}")
    
    if not discharge_df.empty:
        print(f"\nDischarge cycle analysis:")
        for battery in discharge_df['battery'].unique():
            battery_data = discharge_df[discharge_df['battery'] == battery]
            print(f"  {battery}:")
            print(f"    Cycles: {len(battery_data)}")
            if 'capacity_delivered' in battery_data.columns:
                print(f"    Capacity range: {battery_data['capacity_delivered'].min():.3f} - {battery_data['capacity_delivered'].max():.3f} Ah")
            if 'soh' in battery_data.columns:
                soh_data = battery_data['soh'].dropna()
                if not soh_data.empty:
                    print(f"    SoH range: {soh_data.min():.1f}% - {soh_data.max():.1f}%")
            print(f"    Cycle range: {battery_data['cycle_number'].min()} - {battery_data['cycle_number'].max()}")
    
    print(f"\nFiles saved:")
    print(f"  - nasa_discharge_cycles.csv")
    print(f"  - nasa_charge_cycles.csv") 
    print(f"  - nasa_impedance_cycles.csv")
    print(f"  - nasa_battery_summary.csv")
    print(f"  - nasa_battery_comprehensive_analysis.png")
    
    return discharge_df, charge_df, impedance_df

if __name__ == "__main__":
    discharge_df, charge_df, impedance_df = main()
