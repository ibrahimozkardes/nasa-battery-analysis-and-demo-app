#!/usr/bin/env python3
"""
NASA Battery Dataset - Exploratory Data Analysis (EDA)
Keşifsel Veri Analizi ve Özellik Seçimi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def load_and_prepare_data():
    """Veri setlerini yükle ve EDA için hazırla"""
    print("=== Veri Setleri Yükleniyor ===")
    
    # Load processed data
    discharge_df = pd.read_csv('nasa_discharge_cycles.csv')
    charge_df = pd.read_csv('nasa_charge_cycles.csv')
    impedance_df = pd.read_csv('nasa_impedance_cycles.csv')
    
    print(f"Discharge cycles: {len(discharge_df)}")
    print(f"Charge cycles: {len(charge_df)}")
    print(f"Impedance cycles: {len(impedance_df)}")
    
    # Focus on discharge data for SoH/SoC analysis
    df = discharge_df.copy()
    
    # Add derived features for better analysis
    df['voltage_drop'] = df['voltage_start'] - df['voltage_end']
    df['voltage_efficiency'] = df['voltage_end'] / df['voltage_start']
    df['energy_efficiency'] = df['energy_delivered'] / (df['voltage_start'] * abs(df['current_mean']) * df['discharge_duration'] / 3600)
    df['capacity_retention'] = df['capacity_delivered'] / df.groupby('battery')['capacity_delivered'].transform('max')
    df['temperature_efficiency'] = df['energy_delivered'] / df['temp_rise'].replace(0, np.nan)
    df['power_density'] = df['energy_delivered'] / df['discharge_duration'] * 3600  # W
    
    # Cycle-based features
    df['cycle_normalized'] = df['cycle_number'] / df.groupby('battery')['cycle_number'].transform('max')
    df['degradation_rate'] = df.groupby('battery')['soh'].diff() / df.groupby('battery')['cycle_number'].diff()
    
    return df, charge_df, impedance_df

def statistical_summary_analysis(df):
    """İstatistiksel özet analizi"""
    print("\n=== İSTATİSTİKSEL ÖZET ANALİZİ ===")
    
    # Key features for analysis
    key_features = ['soh', 'capacity_delivered', 'voltage_start', 'voltage_end', 'voltage_drop',
                   'current_mean', 'temp_mean', 'temp_rise', 'energy_delivered', 'discharge_duration',
                   'power_density', 'capacity_retention']
    
    print("\nTemel İstatistikler:")
    summary_stats = df[key_features].describe()
    print(summary_stats.round(3))
    
    # Skewness and kurtosis analysis
    print("\nÇarpıklık (Skewness) ve Basıklık (Kurtosis) Analizi:")
    skew_kurt = pd.DataFrame({
        'Skewness': df[key_features].skew(),
        'Kurtosis': df[key_features].kurtosis()
    })
    print(skew_kurt.round(3))
    
    # Normality tests
    print("\nNormallik Testleri (Shapiro-Wilk p-values):")
    normality_results = {}
    for feature in key_features:
        if len(df[feature].dropna()) > 3:
            sample_size = min(5000, len(df[feature].dropna()))  # Limit sample size for Shapiro-Wilk
            sample = df[feature].dropna().sample(n=sample_size, random_state=42)
            stat, p_value = stats.shapiro(sample)
            normality_results[feature] = p_value
    
    normality_df = pd.DataFrame.from_dict(normality_results, orient='index', columns=['p_value'])
    normality_df['is_normal'] = normality_df['p_value'] > 0.05
    print(normality_df.round(6))
    
    return summary_stats, skew_kurt, normality_df

def correlation_analysis(df):
    """Korelasyon analizi"""
    print("\n=== KORELASYON ANALİZİ ===")
    
    # Select features for correlation analysis (numeric only)
    corr_features = ['soh', 'capacity_delivered', 'voltage_start', 'voltage_end', 'voltage_drop',
                    'voltage_mean', 'voltage_std', 'current_mean', 'temp_mean', 'temp_rise',
                    'energy_delivered', 'discharge_duration', 'cycle_number', 'power_density',
                    'capacity_retention', 'voltage_efficiency', 'cycle_normalized']
    
    # Ensure all features exist and are numeric
    available_features = [col for col in corr_features if col in df.columns]
    df_corr = df[available_features].select_dtypes(include=[np.number])
    
    # Calculate correlation matrix
    correlation_matrix = df_corr.corr()
    
    # Create correlation heatmap
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    sns.heatmap(correlation_matrix, 
                mask=mask,
                annot=True, 
                cmap='RdYlBu_r', 
                center=0,
                square=True,
                fmt='.2f',
                cbar_kws={"shrink": .8})
    
    plt.title('Özellikler Arası Korelasyon Matrisi', fontsize=14, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # High correlation pairs
    print("\nYüksek Korelasyonlu Özellik Çiftleri (|r| > 0.7):")
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_val = correlation_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                high_corr_pairs.append({
                    'Feature_1': correlation_matrix.columns[i],
                    'Feature_2': correlation_matrix.columns[j],
                    'Correlation': corr_val
                })
    
    high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('Correlation', key=abs, ascending=False)
    print(high_corr_df.round(3))
    
    return correlation_matrix, high_corr_df

def soh_relationship_analysis(df):
    """SoH ile diğer özellikler arasındaki ilişki analizi"""
    print("\n=== SOH İLİŞKİ ANALİZİ ===")
    
    # Select only numeric columns for correlation
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_columns]
    
    # SoH correlations
    soh_correlations = df_numeric.corr()['soh'].sort_values(key=abs, ascending=False)
    print("\nSoH ile En Yüksek Korelasyonlu Özellikler:")
    print(soh_correlations.head(10).round(3))
    
    # Create SoH relationship plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('State of Health (SoH) İlişki Analizi', fontsize=16)
    
    # 1. SoH vs Capacity
    axes[0, 0].scatter(df['capacity_delivered'], df['soh'], alpha=0.6, c=df['cycle_number'], cmap='viridis')
    axes[0, 0].set_xlabel('Capacity Delivered (Ah)')
    axes[0, 0].set_ylabel('SoH (%)')
    axes[0, 0].set_title(f'SoH vs Capacity\n(r = {df["capacity_delivered"].corr(df["soh"]):.3f})')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. SoH vs Cycle Number
    for battery in df['battery'].unique():
        battery_data = df[df['battery'] == battery].sort_values('cycle_number')
        axes[0, 1].plot(battery_data['cycle_number'], battery_data['soh'], 'o-', 
                       label=battery, markersize=3, alpha=0.8)
    axes[0, 1].set_xlabel('Cycle Number')
    axes[0, 1].set_ylabel('SoH (%)')
    axes[0, 1].set_title('SoH Degradation Over Cycles')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. SoH vs Voltage Drop
    axes[0, 2].scatter(df['voltage_drop'], df['soh'], alpha=0.6, c=df['temp_mean'], cmap='coolwarm')
    axes[0, 2].set_xlabel('Voltage Drop (V)')
    axes[0, 2].set_ylabel('SoH (%)')
    axes[0, 2].set_title(f'SoH vs Voltage Drop\n(r = {df["voltage_drop"].corr(df["soh"]):.3f})')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. SoH vs Temperature
    axes[1, 0].scatter(df['temp_mean'], df['soh'], alpha=0.6, c=df['energy_delivered'], cmap='plasma')
    axes[1, 0].set_xlabel('Average Temperature (°C)')
    axes[1, 0].set_ylabel('SoH (%)')
    axes[1, 0].set_title(f'SoH vs Temperature\n(r = {df["temp_mean"].corr(df["soh"]):.3f})')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. SoH vs Energy Delivered
    axes[1, 1].scatter(df['energy_delivered'], df['soh'], alpha=0.6, c=df['discharge_duration'], cmap='viridis')
    axes[1, 1].set_xlabel('Energy Delivered (Wh)')
    axes[1, 1].set_ylabel('SoH (%)')
    axes[1, 1].set_title(f'SoH vs Energy\n(r = {df["energy_delivered"].corr(df["soh"]):.3f})')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. SoH Distribution by Battery
    df.boxplot(column='soh', by='battery', ax=axes[1, 2])
    axes[1, 2].set_title('SoH Distribution by Battery')
    axes[1, 2].set_ylabel('SoH (%)')
    
    plt.tight_layout()
    plt.savefig('soh_relationship_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return soh_correlations

def soc_feature_analysis(df):
    """SoC tahmini için önemli özelliklerin analizi"""
    print("\n=== SOC ÖZELLİK ANALİZİ ===")
    
    # Create synthetic SoC proxy using voltage and capacity
    # SoC estimation based on voltage curve and remaining capacity
    df['soc_proxy'] = (df['voltage_end'] - df['voltage_end'].min()) / (df['voltage_start'] - df['voltage_end'].min())
    df['soc_capacity_based'] = df['capacity_delivered'] / df.groupby(['battery', 'cycle_number'])['capacity_delivered'].transform('first')
    
    # SoC relevant features
    soc_features = ['voltage_start', 'voltage_end', 'voltage_mean', 'voltage_drop', 'voltage_efficiency',
                   'current_mean', 'temp_mean', 'discharge_duration', 'energy_delivered', 'power_density']
    
    # SoC proxy correlations - select numeric columns only
    numeric_columns = df[soc_features + ['soc_proxy']].select_dtypes(include=[np.number]).columns
    soc_correlations = df[numeric_columns].corr()['soc_proxy'].sort_values(key=abs, ascending=False)
    print("\nSoC Proxy ile En Yüksek Korelasyonlu Özellikler:")
    print(soc_correlations.head(10).round(3))
    
    # Create SoC analysis plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('State of Charge (SoC) Özellik Analizi', fontsize=16)
    
    # 1. Voltage vs SoC proxy
    axes[0, 0].scatter(df['voltage_end'], df['soc_proxy'], alpha=0.6, c=df['cycle_number'], cmap='viridis')
    axes[0, 0].set_xlabel('End Voltage (V)')
    axes[0, 0].set_ylabel('SoC Proxy')
    axes[0, 0].set_title('Voltage-SoC Relationship')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Current vs SoC proxy
    axes[0, 1].scatter(df['current_mean'], df['soc_proxy'], alpha=0.6, c=df['temp_mean'], cmap='coolwarm')
    axes[0, 1].set_xlabel('Average Current (A)')
    axes[0, 1].set_ylabel('SoC Proxy')
    axes[0, 1].set_title('Current-SoC Relationship')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Power density vs SoC proxy
    axes[1, 0].scatter(df['power_density'], df['soc_proxy'], alpha=0.6, c=df['soh'], cmap='plasma')
    axes[1, 0].set_xlabel('Power Density (W)')
    axes[1, 0].set_ylabel('SoC Proxy')
    axes[1, 0].set_title('Power-SoC Relationship')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Temperature vs SoC proxy
    axes[1, 1].scatter(df['temp_mean'], df['soc_proxy'], alpha=0.6, c=df['voltage_drop'], cmap='viridis')
    axes[1, 1].set_xlabel('Temperature (°C)')
    axes[1, 1].set_ylabel('SoC Proxy')
    axes[1, 1].set_title('Temperature-SoC Relationship')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('soc_feature_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return soc_correlations

def temperature_current_analysis(df):
    """Sıcaklık ve akım yükü özel analizi"""
    print("\n=== SICAKLIK VE AKIM YÜKÜ ANALİZİ ===")
    
    # Temperature analysis
    temp_stats = df.groupby('battery').agg({
        'temp_mean': ['mean', 'std', 'min', 'max'],
        'temp_rise': ['mean', 'std'],
        'ambient_temperature': ['mean', 'std']
    }).round(3)
    
    print("\nSıcaklık İstatistikleri (Batarya Bazında):")
    print(temp_stats)
    
    # Current load analysis
    current_stats = df.groupby('battery').agg({
        'current_mean': ['mean', 'std', 'min', 'max'],
        'power_density': ['mean', 'std']
    }).round(3)
    
    print("\nAkım Yükü İstatistikleri (Batarya Bazında):")
    print(current_stats)
    
    # Create temperature-current analysis plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Sıcaklık ve Akım Yükü Analizi', fontsize=16)
    
    # 1. Temperature rise vs SoH
    axes[0, 0].scatter(df['temp_rise'], df['soh'], alpha=0.6, c=df['cycle_number'], cmap='viridis')
    axes[0, 0].set_xlabel('Temperature Rise (°C)')
    axes[0, 0].set_ylabel('SoH (%)')
    axes[0, 0].set_title(f'Temp Rise vs SoH\n(r = {df["temp_rise"].corr(df["soh"]):.3f})')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Current vs SoH
    axes[0, 1].scatter(df['current_mean'], df['soh'], alpha=0.6, c=df['temp_mean'], cmap='coolwarm')
    axes[0, 1].set_xlabel('Average Current (A)')
    axes[0, 1].set_ylabel('SoH (%)')
    axes[0, 1].set_title(f'Current vs SoH\n(r = {df["current_mean"].corr(df["soh"]):.3f})')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Temperature vs Current
    axes[0, 2].scatter(df['temp_mean'], df['current_mean'], alpha=0.6, c=df['energy_delivered'], cmap='plasma')
    axes[0, 2].set_xlabel('Average Temperature (°C)')
    axes[0, 2].set_ylabel('Average Current (A)')
    axes[0, 2].set_title(f'Temp vs Current\n(r = {df["temp_mean"].corr(df["current_mean"]):.3f})')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Temperature evolution over cycles
    for battery in df['battery'].unique():
        battery_data = df[df['battery'] == battery].sort_values('cycle_number')
        axes[1, 0].plot(battery_data['cycle_number'], battery_data['temp_mean'], 
                       'o-', label=battery, markersize=2, alpha=0.8)
    axes[1, 0].set_xlabel('Cycle Number')
    axes[1, 0].set_ylabel('Average Temperature (°C)')
    axes[1, 0].set_title('Temperature Evolution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Current load evolution
    for battery in df['battery'].unique():
        battery_data = df[df['battery'] == battery].sort_values('cycle_number')
        axes[1, 1].plot(battery_data['cycle_number'], battery_data['current_mean'], 
                       'o-', label=battery, markersize=2, alpha=0.8)
    axes[1, 1].set_xlabel('Cycle Number')
    axes[1, 1].set_ylabel('Average Current (A)')
    axes[1, 1].set_title('Current Load Evolution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Power density distribution
    df.boxplot(column='power_density', by='battery', ax=axes[1, 2])
    axes[1, 2].set_title('Power Density Distribution')
    axes[1, 2].set_ylabel('Power Density (W)')
    
    plt.tight_layout()
    plt.savefig('temperature_current_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return temp_stats, current_stats

def feature_importance_analysis(df):
    """Özellik önem analizi"""
    print("\n=== ÖZELLİK ÖNEM ANALİZİ ===")
    
    # Prepare features for importance analysis
    feature_columns = ['voltage_start', 'voltage_end', 'voltage_mean', 'voltage_std', 'voltage_drop',
                      'current_mean', 'temp_mean', 'temp_rise', 'energy_delivered', 'discharge_duration',
                      'power_density', 'cycle_number', 'cycle_normalized', 'voltage_efficiency']
    
    X = df[feature_columns].fillna(df[feature_columns].mean())
    y_soh = df['soh'].fillna(df['soh'].mean())
    y_capacity = df['capacity_delivered'].fillna(df['capacity_delivered'].mean())
    
    # Mutual Information for SoH prediction
    mi_soh = mutual_info_regression(X, y_soh, random_state=42)
    mi_soh_df = pd.DataFrame({
        'Feature': feature_columns,
        'MI_SoH': mi_soh
    }).sort_values('MI_SoH', ascending=False)
    
    print("\nSoH Prediction için Mutual Information Scores:")
    print(mi_soh_df.round(4))
    
    # Mutual Information for Capacity prediction
    mi_capacity = mutual_info_regression(X, y_capacity, random_state=42)
    mi_capacity_df = pd.DataFrame({
        'Feature': feature_columns,
        'MI_Capacity': mi_capacity
    }).sort_values('MI_Capacity', ascending=False)
    
    print("\nCapacity Prediction için Mutual Information Scores:")
    print(mi_capacity_df.round(4))
    
    # Feature importance visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # SoH feature importance
    axes[0].barh(mi_soh_df['Feature'], mi_soh_df['MI_SoH'])
    axes[0].set_xlabel('Mutual Information Score')
    axes[0].set_title('SoH Prediction - Feature Importance')
    axes[0].grid(True, alpha=0.3)
    
    # Capacity feature importance
    axes[1].barh(mi_capacity_df['Feature'], mi_capacity_df['MI_Capacity'])
    axes[1].set_xlabel('Mutual Information Score')
    axes[1].set_title('Capacity Prediction - Feature Importance')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('feature_importance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return mi_soh_df, mi_capacity_df

def dimensionality_analysis(df):
    """Boyut indirgeme analizi (PCA)"""
    print("\n=== BOYUT İNDİRGEME ANALİZİ (PCA) ===")
    
    # Prepare features
    feature_columns = ['voltage_start', 'voltage_end', 'voltage_mean', 'voltage_std', 'voltage_drop',
                      'current_mean', 'temp_mean', 'temp_rise', 'energy_delivered', 'discharge_duration',
                      'power_density', 'cycle_normalized', 'voltage_efficiency']
    
    X = df[feature_columns].fillna(df[feature_columns].mean())
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # Explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    print(f"\nİlk 5 bileşenin açıkladığı varyans:")
    for i in range(5):
        print(f"PC{i+1}: {explained_variance_ratio[i]:.3f} ({cumulative_variance[i]:.3f} kümülatif)")
    
    # PCA visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Scree plot
    axes[0].plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, 'bo-')
    axes[0].set_xlabel('Principal Component')
    axes[0].set_ylabel('Explained Variance Ratio')
    axes[0].set_title('Scree Plot')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Cumulative variance
    axes[1].plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'ro-')
    axes[1].axhline(y=0.95, color='black', linestyle='--', alpha=0.7, label='95% Variance')
    axes[1].set_xlabel('Number of Components')
    axes[1].set_ylabel('Cumulative Explained Variance')
    axes[1].set_title('Cumulative Variance Plot')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. PC1 vs PC2 colored by SoH
    scatter = axes[2].scatter(X_pca[:, 0], X_pca[:, 1], c=df['soh'], cmap='viridis', alpha=0.6)
    axes[2].set_xlabel(f'PC1 ({explained_variance_ratio[0]:.1%} variance)')
    axes[2].set_ylabel(f'PC2 ({explained_variance_ratio[1]:.1%} variance)')
    axes[2].set_title('PCA Visualization (Colored by SoH)')
    plt.colorbar(scatter, ax=axes[2], label='SoH (%)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pca_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Principal component loadings
    loadings = pd.DataFrame(
        pca.components_[:5].T,
        columns=[f'PC{i+1}' for i in range(5)],
        index=feature_columns
    )
    
    print("\nİlk 5 Ana Bileşenin Yüklemeleri:")
    print(loadings.round(3))
    
    return pca, loadings, explained_variance_ratio

def create_feature_selection_report(df, correlation_matrix, mi_soh_df, mi_capacity_df):
    """Özellik seçimi raporu oluştur"""
    print("\n=== ÖZELLİK SEÇİMİ RAPORU ===")
    
    # Define feature categories
    feature_categories = {
        'Voltage_Features': ['voltage_start', 'voltage_end', 'voltage_mean', 'voltage_std', 'voltage_drop', 'voltage_efficiency'],
        'Current_Features': ['current_mean'],
        'Temperature_Features': ['temp_mean', 'temp_rise', 'ambient_temperature'],
        'Energy_Features': ['energy_delivered', 'power_density'],
        'Time_Features': ['discharge_duration', 'cycle_number', 'cycle_normalized'],
        'Capacity_Features': ['capacity_delivered', 'capacity_retention'],
        'Target_Features': ['soh']
    }
    
    print("\n1. ÖZELLİK KATEGORİLERİ:")
    for category, features in feature_categories.items():
        available_features = [f for f in features if f in df.columns]
        print(f"   {category}: {len(available_features)} özellik")
        print(f"     {', '.join(available_features)}")
    
    print("\n2. SOH TAHMİNİ İÇİN ÖNERİLEN ÖZELLİKLER:")
    
    # Top features for SoH prediction
    top_soh_features = mi_soh_df.head(8)['Feature'].tolist()
    print("\n   Mutual Information Tabanlı (Top 8):")
    for i, feature in enumerate(top_soh_features, 1):
        mi_score = mi_soh_df[mi_soh_df['Feature'] == feature]['MI_SoH'].iloc[0]
        corr_with_soh = correlation_matrix.loc[feature, 'soh'] if feature in correlation_matrix.index else 'N/A'
        print(f"     {i}. {feature} (MI: {mi_score:.3f}, Corr: {corr_with_soh:.3f})")
    
    print("\n3. SOC TAHMİNİ İÇİN ÖNERİLEN ÖZELLİKLER:")
    
    # Features most relevant for SoC estimation
    soc_relevant_features = ['voltage_end', 'voltage_mean', 'voltage_drop', 'current_mean', 
                           'discharge_duration', 'energy_delivered', 'power_density', 'temp_mean']
    
    print("\n   SoC için Kritik Özellikler:")
    for i, feature in enumerate(soc_relevant_features, 1):
        if feature in correlation_matrix.index:
            # Correlation with voltage_end (SoC proxy)
            corr_voltage = correlation_matrix.loc[feature, 'voltage_end']
            print(f"     {i}. {feature} (Voltage corr: {corr_voltage:.3f})")
    
    print("\n4. YÜKSEK KORELASYONLU ÖZELLİK ÇİFTLERİ (Multicollinearity):")
    high_corr_threshold = 0.8
    high_corr_pairs = []
    
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_val = correlation_matrix.iloc[i, j]
            if abs(corr_val) > high_corr_threshold:
                high_corr_pairs.append({
                    'Feature_1': correlation_matrix.columns[i],
                    'Feature_2': correlation_matrix.columns[j],
                    'Correlation': corr_val
                })
    
    if high_corr_pairs:
        high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('Correlation', key=abs, ascending=False)
        print("   Dikkat edilmesi gereken yüksek korelasyonlar:")
        for _, row in high_corr_df.iterrows():
            print(f"     {row['Feature_1']} <-> {row['Feature_2']}: {row['Correlation']:.3f}")
    else:
        print("   ✅ Kritik multicollinearity sorunu yok")
    
    print("\n5. FİNAL ÖZELLİK ÖNERİLERİ:")
    
    print("\n   SoH Prediction Model için:")
    soh_final_features = ['cycle_number', 'capacity_delivered', 'voltage_drop', 'energy_delivered', 
                         'temp_rise', 'voltage_mean', 'discharge_duration', 'power_density']
    for i, feature in enumerate(soh_final_features, 1):
        print(f"     {i}. {feature}")
    
    print(f"\n   SoC Estimation Model için:")
    soc_final_features = ['voltage_end', 'voltage_mean', 'current_mean', 'discharge_duration',
                         'temp_mean', 'energy_delivered', 'power_density']
    for i, feature in enumerate(soc_final_features, 1):
        print(f"     {i}. {feature}")
    
    print(f"\n6. ÖZELLİK MÜHENDİSLİĞİ ÖNERİLERİ:")
    print("   ✅ voltage_drop = voltage_start - voltage_end")
    print("   ✅ voltage_efficiency = voltage_end / voltage_start") 
    print("   ✅ power_density = energy_delivered / discharge_duration * 3600")
    print("   ✅ cycle_normalized = cycle_number / max_cycle_per_battery")
    print("   ✅ capacity_retention = capacity_delivered / initial_capacity")
    
    return soh_final_features, soc_final_features

def main():
    """Ana EDA fonksiyonu"""
    print("NASA BATTERY DATASET - EXPLORATORY DATA ANALYSIS")
    print("="*70)
    
    # Load and prepare data
    df, charge_df, impedance_df = load_and_prepare_data()
    
    # Statistical summary analysis
    summary_stats, skew_kurt, normality_df = statistical_summary_analysis(df)
    
    # Correlation analysis
    correlation_matrix, high_corr_df = correlation_analysis(df)
    
    # SoH relationship analysis
    soh_correlations = soh_relationship_analysis(df)
    
    # SoC feature analysis
    soc_correlations = soc_feature_analysis(df)
    
    # Temperature and current analysis
    temp_stats, current_stats = temperature_current_analysis(df)
    
    # Feature importance analysis
    mi_soh_df, mi_capacity_df = feature_importance_analysis(df)
    
    # Dimensionality analysis
    pca, loadings, explained_variance = dimensionality_analysis(df)
    
    # Feature selection report
    soh_features, soc_features = create_feature_selection_report(df, correlation_matrix, mi_soh_df, mi_capacity_df)
    
    # Save EDA results
    print(f"\n=== SONUÇLAR KAYDEDİLİYOR ===")
    
    # Save correlation matrix
    correlation_matrix.to_csv('correlation_matrix.csv')
    
    # Save feature importance results
    mi_soh_df.to_csv('soh_feature_importance.csv', index=False)
    mi_capacity_df.to_csv('capacity_feature_importance.csv', index=False)
    
    # Save PCA results
    pd.DataFrame(loadings).to_csv('pca_loadings.csv')
    
    # Save enhanced dataset
    df.to_csv('enhanced_discharge_data.csv', index=False)
    
    print("✅ EDA tamamlandı! Kaydedilen dosyalar:")
    print("   - correlation_heatmap.png")
    print("   - soh_relationship_analysis.png") 
    print("   - soc_feature_analysis.png")
    print("   - temperature_current_analysis.png")
    print("   - feature_importance_analysis.png")
    print("   - pca_analysis.png")
    print("   - correlation_matrix.csv")
    print("   - soh_feature_importance.csv")
    print("   - capacity_feature_importance.csv")
    print("   - pca_loadings.csv")
    print("   - enhanced_discharge_data.csv")
    
    print(f"\n" + "="*70)
    print("EDA TAMAMLANDI - MODEL GELİŞTİRME İÇİN HAZIR")
    print("="*70)
    
    return df, correlation_matrix, mi_soh_df, mi_capacity_df, soh_features, soc_features

if __name__ == "__main__":
    df, correlation_matrix, mi_soh_df, mi_capacity_df, soh_features, soc_features = main()
