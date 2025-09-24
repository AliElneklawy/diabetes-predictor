# Diabetes Predictor

A web application for predicting diabetes risk based on various health factors.

## Project Structure

```
diabetes-predictor/
├── data/                    # Data files
│   ├── diabetes.csv        # Main dataset
│   └── test.png           # Test image
├── models/                 # Trained models
│   ├── final_model.pkl    # Gradient boosting model
│   └── neural_network/    # Neural network model files
├── notebooks/             # Jupyter notebooks
│   ├── diabetes_prediction.ipynb      # Traditional ML approach
│   └── diabetes_prediction_nn.ipynb   # Neural network approach
└── src/                   # Source code
    ├── app.py            # Flask web application
    ├── make_your_predictions.py      # Command-line prediction script
    ├── static/           # Static assets
    │   └── css/         # CSS styles
    └── templates/       # HTML templates
```

## Features Used for Prediction

- Gender: Impact on diabetes susceptibility
- Age: Range 0-80 years
- Hypertension: Blood pressure condition
- Heart disease: Associated medical condition
- Smoking history: Risk factor
- BMI: Body Mass Index (range 10.16-71.55)
- HbA1c level: Average blood sugar level (past 2-3 months)
- Blood glucose level: Current blood glucose measurement

## Model Performance

### Gradient Boosting Model
- Accuracy: 0.97
- Precision: 0.98
- Recall: 0.69
- F1-score: 0.82
- ROC AUC: 0.85

### Neural Network
- Precision: 0.85
- Recall: 0.72

## Setup and Installation

### Local Installation with uv

1. Clone the repository:
   ```bash
   git clone https://github.com/AliElneklawy/diabetes-predictor.git
   cd diabetes-predictor
   ```

2. Install uv if you haven't already:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. Create a virtual environment and install dependencies:
   ```bash
   uv venv
   uv pip install .
   ```

### Using Docker

1. Build the Docker image:
   ```bash
   docker build -t diabetes-predictor .
   ```

2. Run the container:
   ```bash
   docker run -d -p 5000:80 diabetes-predictor:latest
   ```

The application will be available at `http://localhost:5000`

Or you can just pull the container and run it:
```bash
docker pull alielneklawy/diabetes-predictor:v1
```


## Usage

### Web Application
1. Start the Flask application:
   ```bash
   cd src
   uv run app.py
   ```
2. Open your web browser and navigate to `http://localhost:5000`
3. Fill in the patient's health information in the form
4. Click "Predict" to see the diabetes risk assessment

### Command Line Interface
You can also use the command-line interface:
```bash
uv run src/make_your_predictions.py
```

### Development
- The notebooks in the `notebooks/` directory contain the model development process
- Trained models are saved in the `models/` directory
- Source code is organized in the `src/` directory

## Data

The dataset is located in `data/diabetes.csv` and contains various health metrics used for prediction. The model takes into account:

- Gender
- Age
- Hypertension status
- Heart disease history
- Smoking history
- BMI (Body Mass Index)
- HbA1c level
- Blood glucose level
