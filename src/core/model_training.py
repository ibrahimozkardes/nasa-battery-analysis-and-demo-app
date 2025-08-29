#!/usr/bin/env python3
"""
NASA Battery Dataset - Machine Learning Model Training
SoH (State of Health) ve SoC (State of Charge) tahmin modelleri
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

# Grafik stili ayarla
plt.style.use('default')
sns.set_palette("husl")

class BatteryMLPipeline:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.results = {}
        
    def load_and_prepare_data(self):
        """Veri setlerini yükle ve model eğitimi için hazırla"""
        print("🔋 Veri setleri yükleniyor...")
        
        # Ana veri setini yükle
        self.discharge_df = pd.read_csv('nasa_discharge_cycles.csv')
        print(f"Toplam kayıt sayısı: {len(self.discharge_df)}")
        print(f"Batarya sayısı: {self.discharge_df['battery'].nunique()}")
        
        # Temel istatistikler
        print(f"\nSoH Aralığı: {self.discharge_df['soh'].min():.1f}% - {self.discharge_df['soh'].max():.1f}%")
        print(f"Kapasite Aralığı: {self.discharge_df['capacity_delivered'].min():.3f} - {self.discharge_df['capacity_delivered'].max():.3f} Ah")
        
        return self.discharge_df
    
    def feature_engineering(self, df):
        """Ek özellik mühendisliği ve türetilmiş özellikler"""
        print("\n🔧 Özellik mühendisliği yapılıyor...")
        
        df_enhanced = df.copy()
        
        # Mevcut türetilmiş özellikler kontrol et
        if 'voltage_drop' not in df_enhanced.columns:
            df_enhanced['voltage_drop'] = df_enhanced['voltage_start'] - df_enhanced['voltage_end']
        if 'voltage_efficiency' not in df_enhanced.columns:
            df_enhanced['voltage_efficiency'] = df_enhanced['voltage_end'] / df_enhanced['voltage_start']
        
        # Yeni türetilmiş özellikler
        df_enhanced['energy_efficiency'] = df_enhanced['energy_delivered'] / (
            df_enhanced['voltage_start'] * abs(df_enhanced['current_mean']) * df_enhanced['discharge_duration'] / 3600
        )
        df_enhanced['capacity_retention'] = df_enhanced.groupby('battery')['capacity_delivered'].transform(
            lambda x: x / x.max()
        )
        df_enhanced['temperature_efficiency'] = df_enhanced['energy_delivered'] / df_enhanced['temp_rise'].replace(0, np.nan)
        df_enhanced['power_density'] = df_enhanced['energy_delivered'] / df_enhanced['discharge_duration'] * 3600
        df_enhanced['cycle_normalized'] = df_enhanced.groupby('battery')['cycle_number'].transform(
            lambda x: (x - x.min()) / (x.max() - x.min())
        )
        
        # Batarya yaşlandırma trendi
        df_enhanced['degradation_rate'] = df_enhanced.groupby('battery')['capacity_delivered'].transform(
            lambda x: x.diff() / x.shift(1)
        ).fillna(0)
        
        # SoC proxy features
        df_enhanced['soc_proxy'] = (df_enhanced['voltage_end'] - df_enhanced['voltage_end'].min()) / (
            df_enhanced['voltage_end'].max() - df_enhanced['voltage_end'].min()
        )
        df_enhanced['soc_capacity_based'] = df_enhanced['capacity_delivered'] / df_enhanced['capacity_delivered'].max()
        
        # Categorik değişkenleri encode et
        df_enhanced = pd.get_dummies(df_enhanced, columns=['battery'], prefix='battery')
        
        # Temizlik
        df_enhanced = df_enhanced.replace([np.inf, -np.inf], np.nan)
        
        # Sadece numerik sütunlar için median ile doldur
        numeric_columns = df_enhanced.select_dtypes(include=[np.number]).columns
        df_enhanced[numeric_columns] = df_enhanced[numeric_columns].fillna(df_enhanced[numeric_columns].median())
        
        # Kategorik sütunlar için mode ile doldur
        categorical_columns = df_enhanced.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            df_enhanced[col] = df_enhanced[col].fillna(df_enhanced[col].mode()[0] if not df_enhanced[col].mode().empty else 'unknown')
        
        print(f"Toplam özellik sayısı: {len(df_enhanced.columns)}")
        
        return df_enhanced
    
    def select_features(self, df):
        """Özellik seçimi ve hedef değişken tanımlama"""
        print("\n🎯 Özellik seçimi yapılıyor...")
        
        # Hedef değişkenler
        target_features = ['soh', 'capacity_delivered', 'soc_proxy', 'soc_capacity_based']
        
        # Model için kullanılacak özellikler
        feature_columns = [
            # Temel ölçümler
            'voltage_start', 'voltage_end', 'voltage_mean', 'voltage_std',
            'current_mean', 'temp_mean', 'temp_rise',
            'energy_delivered', 'discharge_duration', 'cycle_number',
            
            # Türetilmiş özellikler
            'voltage_drop', 'voltage_efficiency', 'energy_efficiency',
            'capacity_retention', 'power_density', 'cycle_normalized',
            'degradation_rate',
            
            # Batarya kategorik değişkenleri
            'battery_B0005', 'battery_B0006', 'battery_B0018'
        ]
        
        # Mevcut sütunları kontrol et
        available_features = [col for col in feature_columns if col in df.columns]
        print(f"Kullanılabilir özellik sayısı: {len(available_features)}")
        
        self.feature_columns = available_features
        self.target_features = target_features
        
        return available_features, target_features
    
    def prepare_datasets(self, df, target='soh'):
        """Eğitim ve test setlerini hazırla"""
        print(f"\n📊 {target} için veri setleri hazırlanıyor...")
        
        X = df[self.feature_columns].copy()
        y = df[target].copy()
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=df['battery_B0005'] + df['battery_B0006'] + df['battery_B0018']
        )
        
        # Ölçeklendirme
        scaler = RobustScaler()  # Outlier'lara karşı daha dayanıklı
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers[target] = scaler
        
        print(f"Eğitim seti: {X_train_scaled.shape}")
        print(f"Test seti: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X_train.columns
    
    def train_models(self, X_train, X_test, y_train, y_test, target='soh'):
        """Çoklu model eğitimi ve karşılaştırma"""
        print(f"\n🤖 {target} için modeller eğitiliyor...")
        
        # Model tanımları
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(kernel='rbf', C=1.0)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"  🔄 {name} eğitiliyor...")
            
            # Model eğitimi
            model.fit(X_train, y_train)
            
            # Tahminler
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Metrikler
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            
            # Cross validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
            cv_mae = -cv_scores.mean()
            cv_std = cv_scores.std()
            
            results[name] = {
                'model': model,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'cv_mae': cv_mae,
                'cv_std': cv_std,
                'y_test_pred': y_test_pred
            }
            
            print(f"    Test MAE: {test_mae:.3f}, Test R²: {test_r2:.3f}")
        
        self.models[target] = results
        return results
    
    def hyperparameter_tuning(self, X_train, y_train, target='soh'):
        """En iyi modeller için hiperparametre optimizasyonu"""
        print(f"\n⚙️ {target} için hiperparametre optimizasyonu...")
        
        # En iyi modeli seç (test MAE'ye göre)
        best_model_name = min(self.models[target].keys(), 
                             key=lambda k: self.models[target][k]['test_mae'])
        
        print(f"En iyi model: {best_model_name}")
        
        if best_model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
            base_model = RandomForestRegressor(random_state=42)
            
        elif best_model_name == 'XGBoost':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2]
            }
            base_model = xgb.XGBRegressor(random_state=42)
            
        elif best_model_name == 'Gradient Boosting':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2]
            }
            base_model = GradientBoostingRegressor(random_state=42)
            
        else:
            print("Bu model için hiperparametre optimizasyonu yapılmayacak.")
            return self.models[target][best_model_name]['model']
        
        # Grid Search
        grid_search = GridSearchCV(
            base_model, param_grid, cv=5, 
            scoring='neg_mean_absolute_error', n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"En iyi parametreler: {grid_search.best_params_}")
        print(f"En iyi CV MAE: {-grid_search.best_score_:.3f}")
        
        # Optimize edilmiş modeli sakla
        self.models[target]['optimized'] = {
            'model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_score': -grid_search.best_score_
        }
        
        return grid_search.best_estimator_
    
    def feature_importance_analysis(self, X_train, y_train, feature_names, target='soh'):
        """Özellik önemliliği analizi"""
        print(f"\n🎯 {target} için özellik önemliliği analizi...")
        
        # En iyi modeli al
        if 'optimized' in self.models[target]:
            best_model = self.models[target]['optimized']['model']
        else:
            best_model_name = min(self.models[target].keys(), 
                                 key=lambda k: self.models[target][k]['test_mae'])
            best_model = self.models[target][best_model_name]['model']
        
        # Feature importance
        if hasattr(best_model, 'feature_importances_'):
            importance = best_model.feature_importances_
            feature_imp_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            self.feature_importance[target] = feature_imp_df
            
            # Visualizasyon
            plt.figure(figsize=(10, 8))
            sns.barplot(data=feature_imp_df.head(15), x='importance', y='feature')
            plt.title(f'{target.upper()} - Top 15 Feature Importance')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig(f'{target}_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # CSV olarak kaydet
            feature_imp_df.to_csv(f'{target}_model_feature_importance.csv', index=False)
            
            print(f"Top 5 özellik:")
            print(feature_imp_df.head().to_string(index=False))
    
    def evaluate_and_visualize(self, X_test, y_test, target='soh'):
        """Model performansını değerlendir ve görselleştir"""
        print(f"\n📊 {target} modeli değerlendiriliyor...")
        
        # Sonuçları derle
        results_df = pd.DataFrame(self.models[target]).T
        results_summary = results_df[['test_mae', 'test_rmse', 'test_r2', 'cv_mae']].round(4)
        
        print(f"\n{target.upper()} Model Performans Karşılaştırması:")
        print(results_summary.to_string())
        
        # En iyi modeli belirle
        best_model_name = results_summary['test_mae'].idxmin()
        print(f"\nEn iyi model: {best_model_name}")
        print(f"Test MAE: {results_summary.loc[best_model_name, 'test_mae']}")
        print(f"Test R²: {results_summary.loc[best_model_name, 'test_r2']}")
        
        # Prediction vs Actual plot
        plt.figure(figsize=(15, 5))
        
        # Subplot 1: Prediction vs Actual
        plt.subplot(1, 3, 1)
        y_pred = self.models[target][best_model_name]['y_test_pred']
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel(f'Actual {target.upper()}')
        plt.ylabel(f'Predicted {target.upper()}')
        plt.title(f'{best_model_name} - Prediction vs Actual')
        
        # Subplot 2: Residuals
        plt.subplot(1, 3, 2)
        residuals = y_test - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel(f'Predicted {target.upper()}')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        
        # Subplot 3: Model comparison
        plt.subplot(1, 3, 3)
        mae_scores = [self.models[target][model]['test_mae'] for model in self.models[target].keys()]
        model_names = list(self.models[target].keys())
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
        bars = plt.bar(range(len(model_names)), mae_scores, color=colors)
        plt.xticks(range(len(model_names)), model_names, rotation=45)
        plt.ylabel('Test MAE')
        plt.title('Model Comparison')
        
        # En iyi modeli vurgula
        best_idx = mae_scores.index(min(mae_scores))
        bars[best_idx].set_color('red')
        bars[best_idx].set_alpha(0.8)
        
        plt.tight_layout()
        plt.savefig(f'{target}_model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Sonuçları CSV olarak kaydet
        results_summary.to_csv(f'{target}_model_results.csv')
        
        return results_summary, best_model_name
    
    def save_models(self):
        """Eğitilmiş modelleri kaydet"""
        print("\n💾 Modeller kaydediliyor...")
        
        for target in self.models.keys():
            # En iyi modeli seç
            best_model_name = min(self.models[target].keys(), 
                                 key=lambda k: self.models[target][k]['test_mae'] 
                                 if isinstance(self.models[target][k], dict) and 'test_mae' in self.models[target][k] else float('inf'))
            
            if 'optimized' in self.models[target]:
                best_model = self.models[target]['optimized']['model']
                filename = f'best_{target}_model_optimized.pkl'
            else:
                best_model = self.models[target][best_model_name]['model']
                filename = f'best_{target}_model.pkl'
            
            # Modeli kaydet
            joblib.dump(best_model, filename)
            
            # Scaler'ı kaydet
            scaler_filename = f'{target}_scaler.pkl'
            joblib.dump(self.scalers[target], scaler_filename)
            
            print(f"  ✅ {target} modeli kaydedildi: {filename}")
            print(f"  ✅ {target} scaler kaydedildi: {scaler_filename}")

def main():
    """Ana eğitim pipeline'ı"""
    print("🚀 NASA Battery ML Pipeline Başlatılıyor...\n")
    
    # Pipeline oluştur
    pipeline = BatteryMLPipeline()
    
    # Veri yükleme ve hazırlık
    df = pipeline.load_and_prepare_data()
    df_enhanced = pipeline.feature_engineering(df)
    feature_columns, target_features = pipeline.select_features(df_enhanced)
    
    # Her hedef değişken için model eğitimi
    for target in ['soh', 'capacity_delivered']:
        print(f"\n{'='*60}")
        print(f"🎯 {target.upper()} MODELİ EĞİTİMİ")
        print(f"{'='*60}")
        
        # Veri hazırlama
        X_train, X_test, y_train, y_test, feature_names = pipeline.prepare_datasets(df_enhanced, target)
        
        # Model eğitimi
        results = pipeline.train_models(X_train, X_test, y_train, y_test, target)
        
        # Hiperparametre optimizasyonu
        optimized_model = pipeline.hyperparameter_tuning(X_train, y_train, target)
        
        # Optimize edilmiş modeli tekrar değerlendir
        y_test_pred_opt = optimized_model.predict(X_test)
        opt_mae = mean_absolute_error(y_test, y_test_pred_opt)
        opt_r2 = r2_score(y_test, y_test_pred_opt)
        
        print(f"\n🎉 Optimize edilmiş model performansı:")
        print(f"   Test MAE: {opt_mae:.3f}")
        print(f"   Test R²: {opt_r2:.3f}")
        
        # Özellik önemliliği
        pipeline.feature_importance_analysis(X_train, y_train, feature_names, target)
        
        # Değerlendirme ve görselleştirme
        results_summary, best_model = pipeline.evaluate_and_visualize(X_test, y_test, target)
    
    # Modelleri kaydet
    pipeline.save_models()
    
    print(f"\n🎊 Pipeline tamamlandı!")
    print(f"📁 Çıktı dosyaları:")
    print(f"   • Model dosyaları: best_*_model.pkl")
    print(f"   • Scaler dosyaları: *_scaler.pkl") 
    print(f"   • Sonuç CSVleri: *_model_results.csv")
    print(f"   • Görselleştirmeler: *_model_evaluation.png")

if __name__ == "__main__":
    main()
