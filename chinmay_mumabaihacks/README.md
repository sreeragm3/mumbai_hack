# Hospital Surge Readiness (HSR) Platform

A machine learning-powered platform that predicts hospital resource requirements during high-risk festivals in India, helping healthcare facilities prepare for surge events.

## Features

- **Festival Data Integration**: Fetches festival dates from Calendarific API
- **Historical Analysis**: Generates synthetic historical surge data for training
- **ML Prediction**: Uses Linear Regression to predict blood unit requirements
- **Alert System**: Provides early warnings for upcoming high-risk festivals
- **Actionable Reports**: Generates comprehensive surge readiness reports

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd mumbai_hack
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Create a .env file or set environment variable
export CALENDARIFIC_API_KEY="your_api_key_here"
```

## Usage

Run the main application:
```bash
python main.py
```

The platform will:
1. Fetch festival data for the target year
2. Generate historical surge data
3. Train a machine learning model
4. Generate actionable surge readiness reports

## Configuration

Edit `config.py` to customize:
- Target year for festival data
- High-risk festivals to monitor
- Alert lead times
- Country code

## API Key Setup

1. Get a free API key from [Calendarific](https://calendarific.com/)
2. Set the environment variable:
   - Windows: `set CALENDARIFIC_API_KEY=your_key_here`
   - Linux/Mac: `export CALENDARIFIC_API_KEY=your_key_here`

## Project Structure

```
mumbai_hack/
├── main.py              # Main execution script
├── config.py            # Configuration settings
├── data_fetcher.py      # API integration and data processing
├── model_trainer.py     # ML model training and prediction
├── data_classes.py      # Data structure definitions
├── requirements.txt     # Python dependencies
├── .gitignore          # Git ignore rules
└── data/               # Generated data files (ignored by git)
    ├── festival_calendar.csv
    ├── historical_records.csv
    └── surge_predictor.joblib
```

## Dependencies

- requests==2.31.0
- pandas==2.0.3
- scikit-learn==1.3.0
- joblib==1.3.2
- numpy==1.24.3
- tabulate==0.9.0

## License

This project is licensed under the MIT License.
