# Credit Risk Scoring System

An end-to-end credit risk scoring pipeline built on the **Home Credit Default Risk** dataset. It predicts loan default probabilities and generates industry-standard credit scores.

## Features
- **Microservices Architecture:** Decoupled FastAPI backend and Streamlit frontend dashboard.
- **Workflow Orchestration:** Prefect DAG for automated and resilient pipeline execution.
- **SQL Feature Engineering:** Uses DuckDB to aggregate multi-table relational data (bureau, installments, etc.).
- **Machine Learning:** LightGBM classifier tuned with Optuna for handling imbalanced classes.
- **Credit Scoring (PDO):** Industry-standard Points to Double the Odds (PDO) methodology (Base Score = 650, Base Odds = 11.5:1, PDO = 40, Range = 300–850) with calibrated default probabilities and standard risk tiers.
- **Explainability:** SHAP values for model transparency and reason code generation.
- **CI/CD & Docker:** GitHub actions for automated testing/linting and Dockerized components for easy deployment.

## Quick Start (Local Development)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
pip install -r api/requirements.txt
```

### 2. Run the Pipeline (Prefect DAG)
Instead of running scripts manually, execute the entire automated ML pipeline:
```bash
pip install prefect
python pipeline.py
```

### 3. Launch the Application Locally (via Docker)
To spin up both the FastAPI backend and Streamlit dashboard simultaneously:
```bash
docker compose up --build
```
- The **Dashboard** will be available at: `http://localhost:8501`
- The **API Docs** will be available at: `http://localhost:8000/docs`


## Deployment Guide

Since we upgraded this to a FAANG-level microservice architecture, you can't just host it as a single script anymore. You need to deploy the **API Backend** and the **Streamlit Frontend** separately. Here is how you can do it using free tiers:

### Step 1: Push to GitHub
Ensure all your code, including the `.github/` folder (for your CI/CD pipeline tests), is pushed to a GitHub repository.

### Step 2: Deploy the FastAPI Backend (Render)
We need to deploy the backend API first so it has a public URL.
1. Create a free account on [Render](https://render.com).
2. Click **New +** and select **Web Service**.
3. Connect your GitHub repository.
4. Under the **Runtime** setting, select **Docker**.
5. In the settings, specify the Dockerfile path as `Dockerfile.api`.
6. Click **Create Web Service**. 
7. Once deployed, Render will give you a public URL (e.g., `https://credit-risk-api.onrender.com`). Save this URL.

### Step 3: Deploy the Streamlit Dashboard (Streamlit Cloud)
1. Go to [Streamlit Community Cloud](https://share.streamlit.io).
2. Click **New App** and connect your GitHub repository.
3. Select `app.py` as your main file path.
4. *(Optional)* If you update your `app.py` in the future to make `requests.post()` calls to your API, you can add your Render API URL in the **Advanced Settings** as a Secret.
5. Click **Deploy!**

Alternatively, you can deploy the dashboard on Render exactly like Step 2, but specifying `Dockerfile.app` instead.

## Testing & CI/CD
To run the automated test suite locally:
```bash
pytest tests/
```
Whenever you push code to `main`, GitHub Actions will automatically run these tests and lint the codebase to ensure nothing breaks.
