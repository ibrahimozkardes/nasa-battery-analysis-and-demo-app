#!/usr/bin/env python3
"""
NASA Battery SoH/SoC Prediction REST API
FastAPI ile eğitilmiş modelleri servis haline getirme
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import joblib
import pandas as pd
import numpy as np
import uvicorn
from pathlib import Path
import logging

# Logging kurulumu
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Başlangıç
    logger.info("Loading models on startup...")
    success = load_models()
    if not success:
        logger.error("Failed to load models!")
        raise RuntimeError("Model loading failed")
    yield
    # Kapanış
    logger.info("Shutting down...")

# FastAPI uygulaması
app = FastAPI(
    title="NASA Battery SoH/SoC Prediction API",
    description="Machine Learning API for predicting State of Health (SoH) and battery capacity",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Modeller ve scaler'lar için global değişkenler
models = {}
scalers = {}

class BatteryData(BaseModel):
    """Batarya verisi için input model"""
    voltage_start: float = Field(..., ge=0, le=5, description="Başlangıç voltajı (V)")
    voltage_end: float = Field(..., ge=0, le=5, description="Bitiş voltajı (V)")  
    voltage_mean: float = Field(..., ge=0, le=5, description="Ortalama voltaj (V)")
    voltage_std: float = Field(..., ge=0, le=1, description="Voltaj standart sapması")
    current_mean: float = Field(..., le=0, description="Ortalama akım (A) - deşarj için negatif")
    temp_mean: float = Field(..., ge=0, le=50, description="Ortalama sıcaklık (°C)")
    temp_rise: float = Field(..., ge=0, le=20, description="Sıcaklık artışı (°C)")
    energy_delivered: float = Field(..., ge=0, description="Verilen enerji (Wh)")
    discharge_duration: float = Field(..., ge=0, description="Deşarj süresi (saniye)")
    cycle_number: int = Field(..., ge=1, description="Döngü numarası")
    
    # Batarya tipi (one-hot encoded)
    battery_B0005: bool = Field(False, description="B0005 bataryası mı")
    battery_B0006: bool = Field(False, description="B0006 bataryası mı") 
    battery_B0018: bool = Field(False, description="B0018 bataryası mı")
    
    class Config:
        json_schema_extra = {
            "example": {
                "voltage_start": 4.19,
                "voltage_end": 3.28,
                "voltage_mean": 3.53,
                "voltage_std": 0.24,
                "current_mean": -1.82,
                "temp_mean": 32.5,
                "temp_rise": 9.9,
                "energy_delivered": 6.61,
                "discharge_duration": 3690.0,
                "cycle_number": 1,
                "battery_B0005": True,
                "battery_B0006": False,
                "battery_B0018": False
            }
        }

class PredictionResponse(BaseModel):
    """Tahmin sonucu response model"""
    soh_prediction: float = Field(..., description="Sağlık durumu tahmini (%)")
    capacity_prediction: float = Field(..., description="Kapasite tahmini (Ah)")
    confidence_metrics: Dict = Field(..., description="Güven metrikleri")
    derived_features: Dict = Field(..., description="Hesaplanan türetilmiş özellikler")

class HealthStatus(BaseModel):
    """API sağlık kontrolü response modeli"""
    status: str
    models_loaded: List[str]
    api_version: str

def load_models():
    """Eğitilmiş modelleri ve scaler'ları yükle"""
    try:
        # Temel path'i belirle
        import os
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        models_path = os.path.join(base_path, 'models')
        
        # SoH modeli ve scaler
        models['soh'] = joblib.load(os.path.join(models_path, 'best_soh_model.pkl'))
        scalers['soh'] = joblib.load(os.path.join(models_path, 'soh_scaler.pkl'))
        logger.info("SoH model loaded successfully")
        
        # Kapasite modeli ve scaler
        models['capacity_delivered'] = joblib.load(os.path.join(models_path, 'best_capacity_delivered_model.pkl'))
        scalers['capacity_delivered'] = joblib.load(os.path.join(models_path, 'capacity_delivered_scaler.pkl'))
        logger.info("Capacity model loaded successfully")
        
        return True
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        return False

def calculate_derived_features(data: BatteryData) -> Dict:
    """Türetilmiş özellikleri hesapla"""
    derived = {}
    
    # Voltaj düşüşü ve verimliliği
    derived['voltage_drop'] = data.voltage_start - data.voltage_end
    derived['voltage_efficiency'] = data.voltage_end / data.voltage_start if data.voltage_start > 0 else 0
    
    # Enerji verimliliği
    derived['energy_efficiency'] = (
        data.energy_delivered / (data.voltage_start * abs(data.current_mean) * data.discharge_duration / 3600)
        if data.voltage_start > 0 and data.current_mean != 0 and data.discharge_duration > 0 else 0
    )
    
    # Güç yoğunluğu
    derived['power_density'] = data.energy_delivered / data.discharge_duration * 3600 if data.discharge_duration > 0 else 0
    
    # Sıcaklık verimliliği
    derived['temperature_efficiency'] = data.energy_delivered / data.temp_rise if data.temp_rise > 0 else 0
    
    # Normalize edilmiş döngü (yaklaşım - gerçek normalizasyon için tüm veri seti gerekli)
    derived['cycle_normalized'] = min(data.cycle_number / 500, 1.0)  # Yaklaşım
    
    # Kapasite koruma yaklaşımı (başlangıç kapasitesine göre)
    estimated_initial_capacity = 1.85  # B0005 başlangıç kapasitesi yaklaşımı
    if data.battery_B0006:
        estimated_initial_capacity = 2.03
    elif data.battery_B0018:
        estimated_initial_capacity = 1.86
    
    # Capacity retention estimate
    derived['capacity_retention'] = min(data.energy_delivered / (estimated_initial_capacity * 3.6), 1.0)  # Rough estimate
    
    # Degradation rate proxy
    derived['degradation_rate'] = 0.0  # Cannot calculate without time series
    
    return derived

def prepare_features(data: BatteryData, derived: Dict) -> pd.DataFrame:
    """Model için özellik vektörü hazırla - DataFrame olarak"""
    
    # Feature order (model_training.py'daki sırayla aynı olmalı)
    feature_names = [
        'voltage_start', 'voltage_end', 'voltage_mean', 'voltage_std',
        'current_mean', 'temp_mean', 'temp_rise', 'energy_delivered',
        'discharge_duration', 'cycle_number', 'voltage_drop', 'voltage_efficiency',
        'energy_efficiency', 'capacity_retention', 'power_density', 
        'cycle_normalized', 'degradation_rate', 'battery_B0005', 
        'battery_B0006', 'battery_B0018'
    ]
    
    features_values = [
        data.voltage_start,
        data.voltage_end, 
        data.voltage_mean,
        data.voltage_std,
        data.current_mean,
        data.temp_mean,
        data.temp_rise,
        data.energy_delivered,
        data.discharge_duration,
        data.cycle_number,
        derived['voltage_drop'],
        derived['voltage_efficiency'],
        derived['energy_efficiency'],
        derived['capacity_retention'],
        derived['power_density'],
        derived['cycle_normalized'],
        derived['degradation_rate'],
        float(data.battery_B0005),  # bool -> float
        float(data.battery_B0006),  # bool -> float
        float(data.battery_B0018)   # bool -> float
    ]
    
    # DataFrame oluştur
    df = pd.DataFrame([features_values], columns=feature_names)
    return df

@app.get("/", response_model=HealthStatus)
async def health_check():
    """API sağlık kontrolü"""
    return HealthStatus(
        status="healthy",
        models_loaded=list(models.keys()),
        api_version="1.0.0"
    )

@app.get("/health", response_model=HealthStatus)
async def detailed_health():
    """Detaylı sağlık kontrolü"""
    return HealthStatus(
        status="healthy" if models and scalers else "unhealthy",
        models_loaded=list(models.keys()),
        api_version="1.0.0"
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_battery_health(data: BatteryData):
    """
    Batarya SoH ve kapasite tahmini
    
    - **SoH**: Sağlık durumu yüzdesi (%)
    - **Capacity**: Mevcut kapasite (Ah)
    """
    try:
        # Modellerin yüklü olduğunu kontrol et
        if not models or not scalers:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        # Türetilmiş özellikleri hesapla
        derived_features = calculate_derived_features(data)
        
        # Özellik vektörünü hazırla
        features_df = prepare_features(data, derived_features)
        
        # SoH tahmini
        soh_features_scaled = scalers['soh'].transform(features_df)
        soh_prediction = models['soh'].predict(soh_features_scaled)[0]
        
        # Kapasite tahmini  
        capacity_features_scaled = scalers['capacity_delivered'].transform(features_df)
        capacity_prediction = models['capacity_delivered'].predict(capacity_features_scaled)[0]
        
        # Güven metrikleri (basit metrikler) - bool tiplerini Python bool'a çevir
        confidence_metrics = {
            "voltage_range_check": bool(2.0 <= data.voltage_end <= 4.3),
            "current_range_check": bool(-2.5 <= data.current_mean <= 0),
            "temp_range_check": bool(20 <= data.temp_mean <= 40),
            "soh_range_check": bool(50 <= soh_prediction <= 105),
            "capacity_plausible": bool(0.8 <= capacity_prediction <= 2.5)
        }
        
        # Overall confidence
        confidence_metrics["overall_confidence"] = float(sum(confidence_metrics.values()) / len(confidence_metrics) * 100)
        
        return PredictionResponse(
            soh_prediction=round(float(soh_prediction), 2),
            capacity_prediction=round(float(capacity_prediction), 4),
            confidence_metrics=confidence_metrics,
            derived_features={k: round(v, 4) for k, v in derived_features.items()}
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch")
async def batch_predict(data_list: List[BatteryData]):
    """
    Çoklu batarya verisi için batch tahmin
    """
    try:
        if len(data_list) > 100:
            raise HTTPException(status_code=400, detail="Batch size limit: 100")
        
        results = []
        for data in data_list:
            # Her bir veri için tekil tahmin yap
            derived_features = calculate_derived_features(data)
            features_df = prepare_features(data, derived_features)
            
            # Predictions
            soh_pred = models['soh'].predict(scalers['soh'].transform(features_df))[0]
            cap_pred = models['capacity_delivered'].predict(scalers['capacity_delivered'].transform(features_df))[0]
            
            results.append({
                "soh_prediction": round(float(soh_pred), 2),
                "capacity_prediction": round(float(cap_pred), 4),
                "input_cycle": data.cycle_number
            })
        
        return {"predictions": results, "count": len(results)}
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/model/info")
async def model_info():
    """Model bilgileri"""
    try:
        info = {
            "soh_model": {
                "type": str(type(models['soh']).__name__),
                "features_count": len(scalers['soh'].feature_names_in_) if hasattr(scalers['soh'], 'feature_names_in_') else 20
            },
            "capacity_model": {
                "type": str(type(models['capacity_delivered']).__name__),
                "features_count": len(scalers['capacity_delivered'].feature_names_in_) if hasattr(scalers['capacity_delivered'], 'feature_names_in_') else 20
            }
        }
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model info error: {str(e)}")

if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0", 
        port=8001,
        reload=True,
        log_level="info"
    )
