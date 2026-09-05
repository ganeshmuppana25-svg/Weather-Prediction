# WeatherPredict

A clean Flask + JavaScript demo that uses a **Decision Tree Classifier** to predict one of six weather conditions from 10 weather measurements.

## Run

```bash
python -m pip install -r requirements.txt
python train_model.py
python app.py
```

Open `http://127.0.0.1:5000` in a browser.

## Features
- 10 weather inputs
- Decision Tree prediction with confidence
- Actual model performance metrics
- Dataset information
- Light/Dark mode with saved preference
- Responsive desktop/tablet/mobile layout
- Flask JSON prediction API

## Weather classes
Sunny, Partly Cloudy, Cloudy, Rainy, Thunderstorm, Foggy.
