# FIFA World Cup 2026 AI Match Predictor 🏆

An end-to-end Machine Learning web application that predicts the outcome of the 2026 FIFA World Cup matches using historical match data, player statistics, and team forms. It features a striking **Neo Brutalism** user interface and is fully configured for serverless deployment.

<!-- [INSERT UI SCREENSHOT HERE: e.g., ![Neo Brutalism UI Preview](preview.png)] -->

## ✨ Features

- **XGBoost Machine Learning Engine**: Predicts Match Winner (Home/Away/Draw) with ~70% binary accuracy based on 964 historical World Cup matches and 29 custom features.
- **Poisson-based Score Generator**: Calculates expected goals (xG) and discrete match scores based on attacking and defensive strengths.
- **Deep Match Statistics**: Simulates and outputs realistic in-game statistics including possession, shots, pass accuracy, and even a heuristic Man of the Match.
- **Neo Brutalism Aesthetic**: A raw, high-contrast user interface with bold typography and heavy shadows, providing a highly engaging user experience.
- **Serverless Ready**: Out-of-the-box configuration for instant deployment to Vercel via `@vercel/python`.

<!-- [INSERT PREDICTION RESULTS SCREENSHOT HERE: e.g., ![Prediction Results](results.png)] -->

## 📂 Project Structure

```text
├── data/
│   ├── raw/             # Historical datasets (matches, world cup summaries, rankings)
│   └── processed/       # ML-ready datasets (squads, team profiles, player stats)
├── models/              # Pre-trained XGBoost Model (model.pkl)
├── scripts/             # Data ingestion and model training scripts
├── static/              # Frontend assets (index.html, style.css, app.js)
├── app.py               # Flask REST API backend
├── feature_engineering.py # Data transformation pipeline
├── model.py             # Inference engine wrapper
├── requirements.txt     # Locked dependencies for deployment
└── vercel.json          # Serverless routing config
```

## 🚀 Local Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/world-cup-predictor.git
   cd world-cup-predictor
   ```

2. **Create a virtual environment (Optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask server**
   ```bash
   python app.py
   ```
   The application will be accessible at `http://localhost:5000`.

## 🧠 Retraining the Model

If you wish to update the datasets or retrain the ML model as real 2026 World Cup data emerges:

1. Process new data:
   ```bash
   python scripts/data_collector.py
   ```
2. Re-train the XGBoost model:
   ```bash
   python scripts/train_model.py
   ```
   This will automatically evaluate the model, log accuracy metrics, and overwrite `models/model.pkl`.

## ☁️ Deployment (Vercel)

This project is configured out-of-the-box for serverless deployment.

1. Create a GitHub repository and push your local code to it.
2. Log into [Vercel](https://vercel.com/) and click **Add New > Project**.
3. Import your newly created GitHub repository.
4. Vercel will automatically detect the `vercel.json` file and deploy the Flask API and frontend.
5. Click **Deploy**.

<!-- [INSERT VERCEL DEPLOYMENT SUCCESS SCREENSHOT HERE: e.g., ![Deployed App](deployed.png)] -->

## 🛠️ Technology Stack

- **Frontend**: Vanilla HTML5, CSS3 (Neo Brutalism), JavaScript (Fetch API)
- **Backend API**: Python, Flask
- **Machine Learning**: XGBoost, Scikit-Learn, Pandas, NumPy
- **Deployment**: Vercel Serverless Functions
