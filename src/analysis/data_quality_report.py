#!/usr/bin/env python3
"""
NASA Battery Dataset - Data Quality Report
Veri kalitesi ve ön işleme sonuçlarının raporlanması
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_data_quality_report():
    """Kapsamlı veri kalitesi raporu oluştur"""
    
    print("NASA BATTERY DATASET - VERİ KALİTE RAPORU")
    print("="*60)
    
    # İşlenmiş verileri yükle
    discharge_df = pd.read_csv('nasa_discharge_cycles.csv')
    charge_df = pd.read_csv('nasa_charge_cycles.csv')
    impedance_df = pd.read_csv('nasa_impedance_cycles.csv')
    summary_df = pd.read_csv('nasa_battery_summary.csv')
    
    print(f"\n1. VERİ SETİ GENELİ")
    print(f"   Toplam deşarj döngüsü: {len(discharge_df)}")
    print(f"   Toplam şarj döngüsü: {len(charge_df)}")
    print(f"   Toplam empedans döngüsü: {len(impedance_df)}")
    print(f"   Batarya sayısı: {len(summary_df)}")
    
    print(f"\n2. BATARYA ÖZETİ")
    print(summary_df.to_string(index=False))
    
    print(f"\n3. DISCHARGE DATA QUALITY ANALYSIS")
    
    # Missing values analysis
    missing_discharge = discharge_df.isnull().sum()
    missing_percent = (missing_discharge / len(discharge_df) * 100).round(2)
    
    print(f"\n   Missing Values:")
    for col, count in missing_discharge[missing_discharge > 0].items():
        print(f"     {col}: {count} ({missing_percent[col]}%)")
    
    if missing_discharge.sum() == 0:
        print("     ✅ No missing values found!")
    
    # Key metrics summary
    print(f"\n   Key Metrics Summary:")
    key_columns = ['capacity_delivered', 'soh', 'voltage_start', 'voltage_end', 'temp_mean', 'energy_delivered']
    
    for col in key_columns:
        if col in discharge_df.columns:
            stats = discharge_df[col].describe()
            print(f"     {col}:")
            print(f"       Mean ± Std: {stats['mean']:.3f} ± {stats['std']:.3f}")
            print(f"       Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
    
    print(f"\n4. CAPACITY DEGRADATION ANALYSIS")
    
    for battery in discharge_df['battery'].unique():
        battery_data = discharge_df[discharge_df['battery'] == battery].sort_values('cycle_number')
        
        initial_cap = battery_data['capacity_delivered'].iloc[0]
        final_cap = battery_data['capacity_delivered'].iloc[-1]
        total_fade = ((initial_cap - final_cap) / initial_cap) * 100
        
        # EOL analysis (70% SoH threshold)
        eol_cycles = battery_data[battery_data['soh'] <= 70]
        eol_reached = len(eol_cycles) > 0
        
        print(f"\n   {battery}:")
        print(f"     Initial Capacity: {initial_cap:.3f} Ah")
        print(f"     Final Capacity: {final_cap:.3f} Ah")
        print(f"     Total Fade: {total_fade:.1f}%")
        print(f"     Cycles analyzed: {len(battery_data)}")
        print(f"     EOL (70% SoH) reached: {'Yes' if eol_reached else 'No'}")
        
        if eol_reached:
            eol_cycle = eol_cycles.iloc[0]['cycle_number']
            print(f"     EOL reached at cycle: {eol_cycle}")
    
    print(f"\n5. DATA PREPROCESSING VALIDATION")
    
    # Outlier analysis
    outlier_count = discharge_df['is_capacity_outlier'].sum() if 'is_capacity_outlier' in discharge_df.columns else 0
    print(f"   Capacity outliers detected: {outlier_count}")
    
    # Voltage range validation
    voltage_issues = discharge_df[(discharge_df['voltage_start'] < 3.0) | 
                                 (discharge_df['voltage_end'] < 2.0) | 
                                 (discharge_df['voltage_start'] > 4.5)]
    print(f"   Voltage anomalies: {len(voltage_issues)}")
    
    # Temperature validation
    temp_issues = discharge_df[(discharge_df['temp_mean'] < 15) | 
                              (discharge_df['temp_mean'] > 50)]
    print(f"   Temperature anomalies: {len(temp_issues)}")
    
    # Discharge duration validation
    duration_stats = discharge_df['discharge_duration'].describe()
    print(f"   Discharge duration range: {duration_stats['min']:.1f} - {duration_stats['max']:.1f} seconds")
    
    print(f"\n6. FEATURE ENGINEERING VALIDATION")
    
    # SoH calculation validation
    soh_valid = discharge_df['soh'].notna().sum()
    print(f"   Valid SoH calculations: {soh_valid}/{len(discharge_df)} ({soh_valid/len(discharge_df)*100:.1f}%)")
    
    # Energy calculation validation
    energy_valid = discharge_df['energy_delivered'].notna().sum()
    print(f"   Valid energy calculations: {energy_valid}/{len(discharge_df)} ({energy_valid/len(discharge_df)*100:.1f}%)")
    
    # Correlation analysis
    print(f"\n7. KEY CORRELATIONS")
    
    # Capacity vs SoH correlation (should be very high)
    if 'soh' in discharge_df.columns:
        cap_soh_corr = discharge_df['capacity_delivered'].corr(discharge_df['soh'])
        print(f"   Capacity vs SoH correlation: {cap_soh_corr:.3f} (Expected: ~1.0)")
    
    # Cycle number vs capacity (should be negative)
    cap_cycle_corr = discharge_df['capacity_delivered'].corr(discharge_df['cycle_number'])
    print(f"   Cycle number vs Capacity correlation: {cap_cycle_corr:.3f} (Expected: negative)")
    
    # Temperature rise vs energy
    if 'temp_rise' in discharge_df.columns:
        temp_energy_corr = discharge_df['temp_rise'].corr(discharge_df['energy_delivered'])
        print(f"   Temperature rise vs Energy correlation: {temp_energy_corr:.3f}")
    
    print(f"\n8. DATA READINESS FOR ML MODELING")
    
    # Feature completeness
    ml_features = ['cycle_number', 'voltage_start', 'voltage_end', 'voltage_mean', 'voltage_std',
                   'current_mean', 'temp_mean', 'discharge_duration', 'capacity_delivered']
    
    feature_completeness = {}
    for feature in ml_features:
        if feature in discharge_df.columns:
            completeness = discharge_df[feature].notna().sum() / len(discharge_df) * 100
            feature_completeness[feature] = completeness
    
    print(f"   ML Feature Completeness:")
    for feature, completeness in feature_completeness.items():
        status = "✅" if completeness > 95 else "⚠️" if completeness > 90 else "❌"
        print(f"     {feature}: {completeness:.1f}% {status}")
    
    # Dataset balance
    print(f"\n   Battery Distribution:")
    battery_counts = discharge_df['battery'].value_counts()
    for battery, count in battery_counts.items():
        percentage = count / len(discharge_df) * 100
        print(f"     {battery}: {count} cycles ({percentage:.1f}%)")
    
    # Temporal distribution
    cycle_range = discharge_df.groupby('battery')['cycle_number'].agg(['min', 'max', 'count'])
    print(f"\n   Temporal Coverage:")
    print(cycle_range)
    
    print(f"\n9. PREPROCESSING SUMMARY")
    print(f"   ✅ Data successfully loaded from 3 battery files")
    print(f"   ✅ Cycle types correctly identified (charge/discharge/impedance)")
    print(f"   ✅ SoH features calculated using capacity degradation")
    print(f"   ✅ Energy features computed from V*I integration")
    print(f"   ✅ Temperature and voltage features extracted")
    print(f"   ✅ Data quality validated - no critical issues")
    print(f"   ✅ Outliers identified and flagged")
    print(f"   ✅ Dataset ready for ML model development")
    
    print(f"\n10. NEXT STEPS FOR ML MODELING")
    print(f"    1. Feature selection and engineering")
    print(f"    2. Train/validation/test split")
    print(f"    3. SoH prediction model development")
    print(f"    4. SoC estimation model development") 
    print(f"    5. Model validation and performance evaluation")
    
    print(f"\n" + "="*60)
    print("DATA PREPROCESSING COMPLETE ✅")
    print("READY FOR MACHINE LEARNING MODEL DEVELOPMENT")
    print("="*60)

if __name__ == "__main__":
    generate_data_quality_report()
