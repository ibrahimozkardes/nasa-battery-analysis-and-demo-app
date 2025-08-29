const express = require('express');
const axios = require('axios');
const bodyParser = require('body-parser');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;
const API_BASE_URL = 'http://localhost:8001';

// Middleware
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());
app.use(express.static('public'));
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));

// Routes

// Ana sayfa
app.get('/', async (req, res) => {
    try {
        // API health check
        const healthResponse = await axios.get(`${API_BASE_URL}/health`);
        const apiStatus = healthResponse.data;
        
        // Model bilgilerini al
        const modelInfoResponse = await axios.get(`${API_BASE_URL}/model/info`);
        const modelInfo = modelInfoResponse.data;
        
        res.render('index', { 
            apiStatus,
            modelInfo,
            prediction: null,
            error: null,
            inputData: null
        });
    } catch (error) {
        console.error('API connection error:', error.message);
        res.render('index', { 
            apiStatus: { status: 'unhealthy', error: error.message },
            modelInfo: null,
            prediction: null,
            error: 'API connection failed'
        });
    }
});

// Tahmin endpoint'i
app.post('/predict', async (req, res) => {
    try {
        const {
            voltage_start, voltage_end, voltage_mean, voltage_std,
            current_mean, temp_mean, temp_rise, energy_delivered,
            discharge_duration, cycle_number, battery_type
        } = req.body;
        
        // Batarya tipini one-hot encoding'e çevir
        const batteryData = {
            voltage_start: parseFloat(voltage_start),
            voltage_end: parseFloat(voltage_end),
            voltage_mean: parseFloat(voltage_mean),
            voltage_std: parseFloat(voltage_std),
            current_mean: parseFloat(current_mean),
            temp_mean: parseFloat(temp_mean),
            temp_rise: parseFloat(temp_rise),
            energy_delivered: parseFloat(energy_delivered),
            discharge_duration: parseFloat(discharge_duration),
            cycle_number: parseInt(cycle_number),
            battery_B0005: battery_type === 'B0005',
            battery_B0006: battery_type === 'B0006',
            battery_B0018: battery_type === 'B0018'
        };
        
        // API'ye tahmin isteği gönder
        const predictionResponse = await axios.post(`${API_BASE_URL}/predict`, batteryData);
        const prediction = predictionResponse.data;
        
        // API status ve model info'yu tekrar al
        const healthResponse = await axios.get(`${API_BASE_URL}/health`);
        const apiStatus = healthResponse.data;
        
        const modelInfoResponse = await axios.get(`${API_BASE_URL}/model/info`);
        const modelInfo = modelInfoResponse.data;
        
        res.render('index', { 
            apiStatus,
            modelInfo,
            prediction,
            error: null,
            inputData: req.body
        });
        
    } catch (error) {
        console.error('Prediction error:', error.message);
        
        // Hata durumunda da temel bilgileri göster
        try {
            const healthResponse = await axios.get(`${API_BASE_URL}/health`);
            const apiStatus = healthResponse.data;
            
            const modelInfoResponse = await axios.get(`${API_BASE_URL}/model/info`);
            const modelInfo = modelInfoResponse.data;
            
            res.render('index', { 
                apiStatus,
                modelInfo,
                prediction: null,
                error: error.response?.data?.detail || error.message,
                inputData: req.body
            });
        } catch (healthError) {
            res.render('index', { 
                apiStatus: { status: 'unhealthy' },
                modelInfo: null,
                prediction: null,
                error: error.response?.data?.detail || error.message,
                inputData: req.body
            });
        }
    }
});

// Batch tahmin endpoint'i
app.post('/predict-batch', async (req, res) => {
    try {
        const batchData = req.body.batch_data;
        
        if (!Array.isArray(batchData) || batchData.length === 0) {
            throw new Error('Invalid batch data');
        }
        
        // Batch verisini API formatına çevir
        const formattedBatchData = batchData.map(item => ({
            voltage_start: parseFloat(item.voltage_start),
            voltage_end: parseFloat(item.voltage_end),
            voltage_mean: parseFloat(item.voltage_mean),
            voltage_std: parseFloat(item.voltage_std),
            current_mean: parseFloat(item.current_mean),
            temp_mean: parseFloat(item.temp_mean),
            temp_rise: parseFloat(item.temp_rise),
            energy_delivered: parseFloat(item.energy_delivered),
            discharge_duration: parseFloat(item.discharge_duration),
            cycle_number: parseInt(item.cycle_number),
            battery_B0005: item.battery_type === 'B0005',
            battery_B0006: item.battery_type === 'B0006',
            battery_B0018: item.battery_type === 'B0018'
        }));
        
        // Batch tahmin isteği
        const batchResponse = await axios.post(`${API_BASE_URL}/predict/batch`, formattedBatchData);
        const batchResults = batchResponse.data;
        
        res.json({
            success: true,
            results: batchResults
        });
        
    } catch (error) {
        console.error('Batch prediction error:', error.message);
        res.status(500).json({
            success: false,
            error: error.response?.data?.detail || error.message
        });
    }
});

// Sample data endpoint
app.get('/api/sample-data', (req, res) => {
    const sampleData = {
        b0005_early: {
            voltage_start: 4.19,
            voltage_end: 3.28,
            voltage_mean: 3.53,
            voltage_std: 0.24,
            current_mean: -1.82,
            temp_mean: 32.5,
            temp_rise: 9.9,
            energy_delivered: 6.61,
            discharge_duration: 3690,
            cycle_number: 1,
            battery_type: 'B0005'
        },
        b0005_late: {
            voltage_start: 4.15,
            voltage_end: 2.8,
            voltage_mean: 3.2,
            voltage_std: 0.35,
            current_mean: -1.75,
            temp_mean: 33.2,
            temp_rise: 11.5,
            energy_delivered: 4.8,
            discharge_duration: 2850,
            cycle_number: 150,
            battery_type: 'B0005'
        },
        b0006_sample: {
            voltage_start: 4.20,
            voltage_end: 2.9,
            voltage_mean: 3.4,
            voltage_std: 0.28,
            current_mean: -1.85,
            temp_mean: 31.8,
            temp_rise: 8.5,
            energy_delivered: 5.2,
            discharge_duration: 3200,
            cycle_number: 100,
            battery_type: 'B0006'
        }
    };
    
    res.json(sampleData);
});

// Hata yakalama middleware
app.use((error, req, res, next) => {
    console.error('Unhandled error:', error);
    res.status(500).render('error', { error: error.message });
});

// 404 handler
app.use((req, res) => {
    res.status(404).render('error', { error: 'Page not found' });
});

// Server başlat
app.listen(PORT, () => {
    console.log(`🚀 NASA Battery Demo App running on http://localhost:${PORT}`);
    console.log(`📡 API Base URL: ${API_BASE_URL}`);
});
