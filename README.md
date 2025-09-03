# 🔋 NASA Battery Health Prediction System

NASA Li-ion pil veri setini kullanarak pil sağlık durumu (SoH) ve kapasite tahminleri yapan makine öğrenmesi sistemi.

## 🚀 Özellikler

- **ML Modelleri**: Random Forest, XGBoost, SVM ile %99+ doğruluk
- **REST API**: FastAPI ile hızlı tahmin servisi  
- **Web Demo**: Bootstrap ile kullanıcı dostu arayüz
- **Docker**: Production-ready containerization

## 📊 Model Performansı

| Model | SoH MAE | Capacity MAE | R² Score |
|-------|---------|--------------|----------|
| **Random Forest** | **0.0004** | **0.0009** | 0.9996 |
| XGBoost | 0.0008 | 0.0012 | 0.9992 |
| SVM | 0.0015 | 0.0018 | 0.9985 |

## 🛠️ Kurulum

### Gereksinimler
- Python 3.12+
- Node.js 18+
- Docker (opsiyonel)

### Hızlı Başlangıç

```bash
# 1. Repo'yu klonlayın
git clone https://github.com/username/nasa-battery-prediction
cd nasa-battery-prediction

# 2. Python environment
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Demo app
cd demo-app && npm install && cd ..

# 4. Modelleri eğitin (gerekirse)
cd src/core && python model_training.py && cd ../..

# 5. Servisleri başlatın
cd src/core && python api_server.py &
cd demo-app && node app.js
```

### Erişim
- **Demo App**: http://localhost:3000 
- **API Docs**: http://localhost:8001/docs

### Docker Deployment

```bash
docker-compose up -d    # Servisleri başlat
docker-compose ps       # Durum kontrol
docker-compose logs -f  # Log takibi
```

## 📱 API Kullanımı

### Temel Endpoints
```bash
# Health check
curl http://localhost:8001/health

# Tahmin
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "battery_type": "B0005",
    "cycle": 100,
    "voltage_measured": 3.2,
    "current_measured": -1.5,
    "temperature_measured": 25.0,
    "current_load": -2.0,
    "voltage_load": 3.1,
    "time": 1800
  }'
```

### Python Örneği

```python
import requests

# Health check
response = requests.get("http://localhost:8001/health")
print(f"Status: {response.json()['status']}")

# Battery prediction
data = {
    "battery_type": "B0005",
    "cycle": 100,
    "voltage_measured": 3.2,
    "current_measured": -1.5,
    "temperature_measured": 25.0,
    "current_load": -2.0,
    "voltage_load": 3.1,
    "time": 1800
}

response = requests.post("http://localhost:8001/predict", json=data)
result = response.json()
print(f"SoH: {result['soh']:.4f}")
print(f"Capacity: {result['capacity_delivered']:.4f} Ah")
```

## 📁 Proje Yapısı

```
nasa-battery-prediction/
├── data/                      # NASA battery dataset files
│   ├── B0005.mat             # Battery cycle data
│   ├── B0006.mat
│   └── B0018.mat
├── demo-app/                  # Node.js web demo
│   ├── app.js
│   └── views/
├── docker/                    # Container configs
│   ├── docker-compose.yml
│   └── Dockerfile.*
├── models/                    # Trained ML models
│   ├── best_soh_model.pkl
│   └── best_capacity_delivered_model.pkl
├── results/                   # Analysis outputs
├── src/
│   ├── analysis/             # Data analysis scripts
│   └── core/                 # Core application
│       ├── api_server.py     # FastAPI server
│       └── model_training.py # ML pipeline
└── requirements.txt
```
```

## 🔬 Teknik Detaylar ve EDA Analizi

### 📊 Veri Seti Özellikleri
- **Toplam Döngü**: 468 deşarj döngüsü
- **Pil Türleri**: 3 farklı Li-ion pil (B0005, B0006, B0018)
- **Özellik Sayısı**: 12 temel + 15 türetilmiş = 27 özellik
- **Veri Boyutu**: ~2.3MB compressed, ~15MB uncompressed
- **Zaman Aralığı**: 616 - 2038 saat arası ölçümler

### 🧪 Feature Engineering
- **Derived Features**: Voltage/current ratios, power calculations
- **Time-based Features**: Cycle progression, temporal patterns  
- **Statistical Features**: Rolling averages, trend analysis
- **Domain-specific**: Coulombic efficiency, energy density

### 📈 Model Eğitim Detayları
```python
# Hyperparameter tuning results
RandomForestRegressor:
    n_estimators: 100
    max_depth: 15
    min_samples_split: 5
    min_samples_leaf: 2
    
XGBoostRegressor:
    n_estimators: 150
    max_depth: 8
    learning_rate: 0.1
    subsample: 0.9
```

## 🐳 Docker ve Production Deployment

### 🏭 Production Features
- **Multi-stage builds** ile optimize edilmiş Docker images
- **Health checks** ve automatic recovery
- **Volume mounting** for persistent model storage
- **Environment-specific configs** (dev/staging/prod)
- **Auto-restart policies** ve resource limits
- **Network isolation** ve security best practices
- **Logging agregation** ve monitoring hooks