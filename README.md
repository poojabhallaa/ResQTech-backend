# 🌍 Res - Q - tech

frontend webpage :- https://resq-tech.vercel.app/
frontend code :- https://github.com/Aarwin10/ResQ-tech
demo video link - https://drive.google.com/drive/folders/1Gg7et9uK6rx3c2Y5XjSE-bTJbRemfSNI5VWoqcLgkNk5-JUU0E9NmH2Mcw9RHR1RldbXUw60

A machine learning-powered application for predicting resource requirements during disaster relief operations.

## Overview

This system uses machine learning models to predict the quantity of essential resources (food, medical aid, and rescue teams) needed during disaster situations based on various factors such as disaster type, severity, location, and weather conditions.

## Features

- **Disaster Data Collection**: Integration with ReliefWeb API to fetch real disaster information
- **Weather Information**: Real-time weather data for the affected region
- **ML-Powered Predictions**: Accurate estimation of resource requirements
- **Visual Map Integration**: Folium maps showing the disaster location and weather conditions
- **AI-Generated Reports**: Comprehensive reports generated using Gemini AI
- **Multi-language Support**: Reports available in multiple languages including English, Hindi, Tamil, and more
- **Image Classification**: Ability to classify disaster types from uploaded images

## Components

1. **Traditional System** (`ml.py`): Simple input form for disaster prediction with report generation
2. **Advanced System** (`ml1.py`): Enhanced version with API integrations, map visualization, and AI summaries
3. **Image Classifier** (`image.py`): Deep learning model for disaster image classification
4. **ML Model Training** (`disaster_relief.ipynb`): Jupyter notebook with model training pipeline

## ML Model Details

The prediction system uses a Random Forest Regressor trained with the following features:
- **Input Features**: disaster_type, location, severity_level, weather_condition, road_blocked, population_affected, medical_cases, infrastructure_damage_percent
- **Output Predictions**: food_required_tons, rescue_teams_required, medical_aid_required
- **Model Pipeline**: Includes categorical feature encoding (OneHotEncoder) and a MultiOutputRegressor with RandomForestRegressor

## Technologies

- **Backend**: Python, Machine Learning (scikit-learn), PyTorch
- **Frontend**: Streamlit
- **APIs**: ReliefWeb, OpenWeatherMap
- **AI**: Google Gemini AI for report generation
- **Data Visualization**: Folium maps, Streamlit components

## Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `streamlit run ml1.py`

## Models

- `disaster_relief_model.pkl`: ML model for resource prediction (Random Forest Regressor)
- `disaster_classifier.pth`: PyTorch model for disaster image classification (ResNet18)

## Dataset

The model was trained on `disaster_relief_dataset.csv` which contains historical data on disaster scenarios and the resources required.

## Future Improvements

- Enhanced disaster detection with real-time data
- Mobile application for field deployment
- Offline functionality for areas with poor connectivity
- Integration with more disaster management APIs
- Model performance improvements (current R² scores indicate room for enhancement)

## License

MIT
