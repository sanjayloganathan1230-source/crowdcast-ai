# 🌆 CrowdCast AI — Crowd Flow Predictor

**AI-Powered Crowd Forecasting & Analytics Platform**

CrowdCast AI is a machine learning-based web application that analyzes historical crowd observations, predicts future crowd levels, identifies peak hours, and visualizes crowd patterns across different locations. It combines Python-based machine learning with an interactive web dashboard to make crowd-flow analysis easier to understand.

> **Note:** This project uses synthetic demo data for educational and demonstration purposes. Predictions are not verified real-world crowd estimates.

## ✨ Features

- **AI Crowd Prediction** — Predict crowd counts for selected locations and future time periods using machine learning.
- **Interactive Dashboard** — Explore crowd statistics, trends, and location comparisons.
- **Peak-Hour Analysis** — Identify periods with higher predicted crowd activity.
- **Location Comparison** — Compare crowd patterns across railway stations, bus terminals, shopping malls, and college campuses.
- **Occupancy Classification** — Categorize predicted occupancy as Low, Moderate, High, or Very High based on configured capacity thresholds.
- **Data Visualization** — View historical observations, prediction trends, and analytics charts.
- **Map Visualization** — Explore demo locations on an interactive map.
- **CSV Export** — Export supported crowd analytics and reports.
- **REST API** — Access application functionality through FastAPI endpoints.
- **Responsive Design** — Use the dashboard on desktop and mobile screens.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas & NumPy | Data processing and numerical operations |
| Scikit-learn | Machine learning and crowd prediction |
| Random Forest Regressor | Crowd-count prediction model |
| FastAPI | Backend REST API |
| React | Frontend user interface |
| Vite | Frontend development and build tooling |
| Recharts | Interactive charts and visualizations |
| SQLite | Local database for historical observations |
| Leaflet & OpenStreetMap | Interactive location map |

*Keep only the technologies actually implemented in your project.*

## 🧠 How It Works

1. **Data Generation:** Create a reproducible synthetic dataset containing timestamps, locations, crowd counts, capacity, and time-related features.
2. **Data Processing:** Clean and prepare observations using Python, Pandas, and NumPy.
3. **Model Training:** Train a machine learning model using historical features and evaluate it against a baseline.
4. **Crowd Prediction:** Generate estimated crowd counts for supported locations and selected future times.
5. **Occupancy Analysis:** Compare predictions with configured location capacities to determine occupancy categories.
6. **Dashboard Visualization:** Display forecasts, hourly patterns, peak periods, and location comparisons through the web interface.

## 🏗️ Project Architecture

```text
CrowdCast-AI/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── routes/
│   │   ├── models/
│   │   └── services/
│   ├── tests/
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   └── services/
│   ├── package.json
│   └── vite.config.js
├── data/
├── README.md
└── .gitignore
```

*This is an illustrative structure. Adjust it to match your actual repository files.*

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or later, compatible with your dependencies
- Node.js and npm
- Git

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/crowdcast-ai.git
cd crowdcast-ai
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Set up the Python backend

Open a terminal:

```bash
cd backend
python -m venv venv
```

Activate the virtual environment.

**Windows:**

```powershell
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the API server using the appropriate module path for your project:

```bash
uvicorn app.main:app --reload
```

Open the interactive API documentation at:

`http://127.0.0.1:8000/docs`

### 3. Start the frontend

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite in your terminal.

*The commands above assume your project follows the illustrative folder structure. Update the module path and commands if your implementation differs.*

## 📊 Machine Learning Evaluation

CrowdCast AI uses regression to estimate crowd counts. Model evaluation can include:

- **MAE (Mean Absolute Error):** Average absolute difference between predicted and observed crowd counts.
- **RMSE (Root Mean Squared Error):** Error measure that gives greater weight to larger prediction errors.
- **Baseline Comparison:** Compare the model with a simple historical-average or similar baseline.

Actual evaluation results should be reported here after running the model on a held-out test dataset. Do not add unverified accuracy claims.

## 🔮 Future Enhancements

- Integration with reliable real-world crowd-count datasets.
- Improved forecasting using time-series models.
- Event and holiday awareness.
- Real-time sensor or transport data integration where authorized.
- Alerts for predicted occupancy thresholds.
- Model monitoring and retraining tools.
- Cloud deployment and public API hosting.

## 🎯 Project Objectives

- Apply machine learning to a practical crowd-analysis problem.
- Build an end-to-end Python AI application.
- Connect a trained model to a REST API and frontend dashboard.
- Practice data preprocessing, model evaluation, and data visualization.
- Develop a portfolio project demonstrating AI/ML and full-stack development skills.

## ⚠️ Limitations

- Synthetic data does not represent actual crowd conditions.
- Predictions may not generalize to real locations without representative training data and validation.
- Occupancy categories depend on configured capacity thresholds.
- The application is an educational prototype, not a certified public-safety or emergency-management system.

## 👨‍💻 Author

**Your Name**

AI & Data Science Student | Python Developer | Machine Learning Enthusiast

- GitHub: https://github.com/sanjayloganathan1230-source/crowdcast-ai

## 📄 License

This project can be distributed under the MIT License if you choose to include that license. Add a `LICENSE` file to the repository before claiming that the project is licensed under MIT.
