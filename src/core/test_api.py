#!/usr/bin/env python3
"""
NASA Battery API Test Client
REST API'yi test etmek için örnek istemci
"""

import requests
import json
import time
from typing import Dict, List

class BatteryAPIClient:
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        
    def health_check(self) -> Dict:
        """API sağlık kontrolü"""
        try:
            response = requests.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def predict_single(self, battery_data: Dict) -> Dict:
        """Tekil batarya tahmini"""
        try:
            response = requests.post(
                f"{self.base_url}/predict", 
                json=battery_data,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def predict_batch(self, battery_data_list: List[Dict]) -> Dict:
        """Batch tahmin"""
        try:
            response = requests.post(
                f"{self.base_url}/predict/batch",
                json=battery_data_list,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def get_model_info(self) -> Dict:
        """Model bilgileri"""
        try:
            response = requests.get(f"{self.base_url}/model/info")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

def create_sample_data():
    """Test için örnek batarya verileri oluştur"""
    
    # B0005 örnek verisi (ilk döngü)
    b0005_early = {
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
    
    # B0005 geç döngü (degraded)
    b0005_late = {
        "voltage_start": 4.15,
        "voltage_end": 2.8,
        "voltage_mean": 3.2,
        "voltage_std": 0.35,
        "current_mean": -1.75,
        "temp_mean": 33.2,
        "temp_rise": 11.5,
        "energy_delivered": 4.8,
        "discharge_duration": 2850.0,
        "cycle_number": 150,
        "battery_B0005": True,
        "battery_B0006": False,
        "battery_B0018": False
    }
    
    # B0006 örnek verisi
    b0006_sample = {
        "voltage_start": 4.20,
        "voltage_end": 2.9,
        "voltage_mean": 3.4,
        "voltage_std": 0.28,
        "current_mean": -1.85,
        "temp_mean": 31.8,
        "temp_rise": 8.5,
        "energy_delivered": 5.2,
        "discharge_duration": 3200.0,
        "cycle_number": 100,
        "battery_B0005": False,
        "battery_B0006": True,
        "battery_B0018": False
    }
    
    # B0018 örnek verisi
    b0018_sample = {
        "voltage_start": 4.18,
        "voltage_end": 3.1,
        "voltage_mean": 3.45,
        "voltage_std": 0.26,
        "current_mean": -1.80,
        "temp_mean": 32.0,
        "temp_rise": 10.2,
        "energy_delivered": 5.8,
        "discharge_duration": 3500.0,
        "cycle_number": 80,
        "battery_B0005": False,
        "battery_B0006": False,
        "battery_B0018": True
    }
    
    return {
        "b0005_early": b0005_early,
        "b0005_late": b0005_late, 
        "b0006": b0006_sample,
        "b0018": b0018_sample
    }

def main():
    """API test client ana fonksiyonu"""
    print("🔋 NASA Battery API Test Client")
    print("=" * 50)
    
    client = BatteryAPIClient()
    
    # 1. Health Check
    print("\n1️⃣ API Health Check:")
    health = client.health_check()
    print(f"   Status: {health.get('status', 'error')}")
    if 'models_loaded' in health:
        print(f"   Models: {health['models_loaded']}")
    
    # 2. Model Info
    print("\n2️⃣ Model Information:")
    model_info = client.get_model_info()
    if 'error' not in model_info:
        for model_name, info in model_info.items():
            print(f"   {model_name}: {info['type']} ({info['features_count']} features)")
    else:
        print(f"   Error: {model_info['error']}")
    
    # 3. Single Predictions
    print("\n3️⃣ Single Predictions:")
    sample_data = create_sample_data()
    
    for name, data in sample_data.items():
        print(f"\n   Testing {name}:")
        result = client.predict_single(data)
        
        if 'error' not in result:
            print(f"      SoH: {result['soh_prediction']}%")
            print(f"      Capacity: {result['capacity_prediction']} Ah")
            print(f"      Confidence: {result['confidence_metrics']['overall_confidence']:.1f}%")
            print(f"      Derived Features: voltage_drop={result['derived_features']['voltage_drop']:.3f}V")
        else:
            print(f"      Error: {result['error']}")
        
        time.sleep(0.1)  # Rate limiting
    
    # 4. Batch Prediction
    print("\n4️⃣ Batch Prediction:")
    batch_data = list(sample_data.values())[:2]  # İlk 2 örnek
    
    batch_result = client.predict_batch(batch_data)
    if 'error' not in batch_result:
        print(f"   Processed {batch_result['count']} samples:")
        for i, pred in enumerate(batch_result['predictions']):
            print(f"      Sample {i+1}: SoH={pred['soh_prediction']}%, Cap={pred['capacity_prediction']}Ah")
    else:
        print(f"   Error: {batch_result['error']}")
    
    print(f"\n✅ Test completed!")
    print(f"🌐 API Documentation: http://localhost:8000/docs")

if __name__ == "__main__":
    main()
