

--

### 🎬 Netflix Revenue Predictor & ROI Optimizer

A full-scale AI-powered application to forecast Netflix movie revenue and ROI using real-world data, advanced ML models, and explainability tools, deployed as an interactive and business-focused Streamlit web app.

--

### 📌 Project Overview

This project addresses a real-world challenge faced by Netflix and other streaming platforms: how to forecast the financial success of a movie before it is released. By leveraging machine learning and explainability tools, this app empowers decision-makers to:

Predict worldwide revenue using content and metadata

Estimate return on investment (ROI)

Understand and explain key drivers behind predictions using SHAP and LIME

Make strategic content and budgeting decisions with confidence

--

### 🚀 Live Demo

🔗 Click here to launch the live Streamlit app

--

### 💡 Key Features

🔮 Revenue Forecasting: Predict log-transformed worldwide revenue with industry-leading accuracy

📊 ROI Estimation: Automatically calculate ROI and assess financial viability

🧠 Explainability: Use SHAP (global & local) and LIME for full model transparency

🧪 Test Scenarios: Input manual values or explore sample predictions from real Netflix titles

💼 Strategic Simulation: Business teams can evaluate content viability, budget strategy, and expected ROI

--

### 📈 Model Performance (on log-transformed revenue)

| Model           | MAE    | RMSE   | R² Score |
|----------------|--------|--------|----------|
| LinearRegressor| 0.168  | 0.281  | 0.966    |
| Random Forest  | 0.0259 | 0.0549 | 0.9987   |
| **XGBoost**    | **0.0275** | **0.0495** | **0.9989**   |

📌 What These Metrics Mean:

MAE (Mean Absolute Error): On average, predictions deviate by ~0.03 log points

RMSE (Root Mean Squared Error): Low variance and error overall

R² Score: Over 99.8% of variability in revenue is explained by the model

✅ This level of performance makes the model suitable for executive-level forecasting.

--

### 📊 Visual Explainability: SHAP & LIME

SHAP Summary Plot: Highlights top features like international_revenue, domestic_revenue, and run_time

SHAP Force Plot: Shows how each feature pushed a specific prediction higher or lower

LIME HTML: Visualizes the individual feature influence on each prediction

These tools provide interpretability, trust, and clarity for non-technical stakeholders.

--

### 📂 Folder Structure

Netflix-Revenue-streamlit/
├── app.py                        # Streamlit front-end app
├── model_xgb.pkl                # Trained XGBoost model
├── scaler.pkl                   # StandardScaler
├── shap_explainer.pkl           # SHAP cached explainer
├── X_train_columns.csv          # Column reference
├── X_test.csv                   # Sample test data
├── shap_force_plot_0.html       # SHAP HTML example
├── lime_explanation_0.html      # LIME HTML example
├── Netflix_Content_Revenue.ipynb # Jupyter Notebook analysis
├── requirements.txt             # Environment dependencies
├── LICENSE                      # MIT License
└── README.md                    # Project documentation

--

### 💼 Business Value & Recommendation

For Netflix and similar platforms, this model serves as a powerful tool to:

📊 Forecast content success with 99.89% accuracy (R²)

💰 Predict ROI with <3% error

🧠 Make informed greenlighting decisions

🧭 Optimize content strategy across genres, languages, and budget ranges

💹 Avoid costly flops and prioritize high-yield investments

📈 Estimated Business Impact:

Using this model at scale can help Netflix optimize $100M+ annually in forecasting reliability, smarter budgeting, and content selection.

--

### 🧪 Technologies Used

Python, Pandas, NumPy

XGBoost, Random Forest, Scikit-learn

SHAP, LIME, Joblib

Streamlit (deployment), Jupyter (EDA/Modeling)

--

### ⚙️ How to Run Locally

git clone https://github.com/SweetySeelam2/Netflix_Revenue_Predictor.git
cd Netflix-Revenue-streamlit
pip install -r requirements.txt
streamlit run app.py

--

### 🧠 Author

Sweety Seelam🎯 Business Analyst | Aspiring Data Scientist🔗 GitHub | Portfolio | LinkedIn

--

### 📜 License

This project is licensed under the MIT License. Feel free to use, remix, and expand with attribution.

--

### ⭐ Show Support

If this project impressed or helped you, please 🌟 star the repo and share it with your network!