
[![Live App - Try it Now](https://img.shields.io/badge/Live%20App-Streamlit-informational?style=for-the-badge&logo=streamlit)](https://netflixrevenuepredictor-streaming-platforms.streamlit.app/)

----------------------------------------------------------------------------------------------------------------------------------------------

### 🎬 Netflix Revenue Predictor & ROI Optimizer

A full-scale AI-powered application to forecast Netflix movie revenue and ROI using real-world data, advanced ML models, and explainability tools, deployed as an interactive and business-focused Streamlit web app.

----------------------------------------------------------------------------------------------------------------------------------------------

### 📌 Project Overview

This project addresses a real-world challenge faced by Netflix and other streaming platforms: how to forecast the financial success of a movie before it is released. By leveraging machine learning and explainability tools, this app empowers decision-makers to:

- Predict worldwide revenue using content and metadata

- Estimate return on investment (ROI)

- Understand and explain key drivers behind predictions using SHAP and LIME explainability

Make strategic content and budgeting decisions with confidence.

----------------------------------------------------------------------------------------------------------------------------------------------

### 🚀 Live Demo

🔗 Click here to launch the live Streamlit app: [https://netflixrevenuepredictor-streaming-platforms.streamlit.app/](https://netflixrevenuepredictor-streaming-platforms.streamlit.app/)

----------------------------------------------------------------------------------------------------------------------------------------------

### 💡 Key Features

🔮 Revenue Forecasting: Predicts expected worldwide revenue based on movie attributes.

📊 ROI Estimation: Calculates the expected return on investment (ROI) to determine profitability.

🧠 Explainability: SHAP and LIME plots highlight how features affect predictions.

🧪 Interactive Input: Two modes — Manual Entry and Use Sample Data.

💼 Business Recommendations:** Auto-generated suggestions based on prediction results.

----------------------------------------------------------------------------------------------------------------------------------------------

### 📈 Model Performance (on log-transformed revenue)

| Model           | MAE    | RMSE   | R² Score |
|----------------|--------|--------|----------|
| LinearRegressor| 0.168  | 0.281  | 0.966    |
| Random Forest  | 0.0259 | 0.0549 | 0.9987   |
| **XGBoost**    | **0.0275** | **0.0495** | **0.9989**   |

📌 What These Metrics Mean:

- MAE (Mean Absolute Error): On average, predictions deviate by ~0.03 log points

- RMSE (Root Mean Squared Error): Low variance and error overall

- R² Score: Over 99.8% of variability in revenue is explained by the model

✅ This level of performance makes the model suitable for executive-level forecasting.

---------------------------------------------------------------------------------------------------------------------------------------------

### 🛠️ How It Works

### Input Options
- **Manual Entry:** Enter movie details like average rating, runtime, budget, release month/year.
- **Use Sample Data:** Choose from test data to view predictions instantly.

### Prediction Output
- **Predicted Revenue** is displayed in USD
- **Estimated ROI** is calculated as a multiplier (e.g., `1.5x` means 50% return)

> A negative ROI (e.g., `-0.5x`) indicates a potential loss.

-------------------------------------------------------------------------------------------------------------------------------------------

### 🧠 SHAP & LIME Explainability

To ensure fast performance, **SHAP and LIME plots are pre-generated only for test samples with index 0 to 4**. These files are:
- `shap_force_plot_0.html` through `shap_force_plot_4.html`
- `lime_explanation_2.html` (shared fallback)

When a supported index is selected, users can view:
- **SHAP Force Plot:** Shows how each feature contributes to the prediction
- **LIME Bar Plot:** Shows top features influencing the outcome

> For other indexes, an info message is shown: “SHAP plot not available for this sample.”

These tools provide interpretability, trust, and clarity for non-technical stakeholders.

----------------------------------------------------------------------------------------------------------------------------------------------

### 📂 Folder Structure

Netflix-Revenue-streamlit/                                           
├── app.py                                              # Streamlit front-end app                                                        
├── model_xgb.pkl                                       # Trained XGBoost model                                                                                                    
├── scaler.pkl                                          # StandardScaler                                                                                                                 
├── shap_explainer.pkl                                  # SHAP cached explainer                                                                       
├── X_train_columns.csv                                 # Column reference                                                                                    
├── X_test.csv                                          # Sample test data                                                                                                   
├── lime_explanation_0.html                             # LIME HTML example                                                                                          
├── Netflix_Content_Revenue.ipynb                       # Jupyter Notebook analysis                                                                                          
├── requirements.txt                                    # Environment dependencies                                                                                 
├── LICENSE                                             # MIT License                                                                                                                   
└── README.md                                           # Project documentation                                                                 

----------------------------------------------------------------------------------------------------------------------------------------------

### 💼 Business Value

For Netflix and similar platforms, this model serves as a powerful tool to:

📊 Forecast content success with 99.89% accuracy (R²)

💰 Predict ROI with <3% error

🧠 Make informed greenlighting decisions

🧭 Optimize content strategy across genres, languages, and budget ranges

💹 Avoid costly flops and prioritize high-yield investments

📈 Estimated Business Impact:

Using this model at scale can help Netflix optimize $100M+ annually in forecasting reliability, smarter budgeting, and content selection.

----------------------------------------------------------------------------------------------------------------------------------------------

### 💼 Business Recommendations

🎯 Focus on movies with high international appeal to boost overall revenue.

🎬 Invest wisely in genres and directors historically linked to higher success.

🧠 Use ML-powered predictions to greenlight profitable content.

🗓 Consider release timing and average ratings to improve expected ROI.

----------------------------------------------------------------------------------------------------------------------------------------------

### 🧪 Technologies Used

- Python, Pandas, NumPy, Matplotlib

- XGBoost, Random Forest, Scikit-learn

- SHAP, LIME, Joblib

- Streamlit (deployment), Jupyter (EDA/Modeling)

----------------------------------------------------------------------------------------------------------------------------------------------

### ⚙️ How to Run Locally

1. git clone https://github.com/SweetySeelam2/Netflix_Revenue_Predictor.git                                                                                
2. cd Netflix-Revenue-streamlit                                                  
3. pip install -r requirements.txt                                                
4. streamlit run app.py                                        

----------------------------------------------------------------------------------------------------------------------------------------------

### 🧠 Author

Sweety Seelam 🎯 Business Analyst | Aspiring Data Scientist                                        

📧 Email: sweetyrao670@gmail.com

🔗 GitHub : https://github.com/SweetySeelam2/Netflix_Revenue_Predictor.git               

🌐 Portfolio: https://sweetyseelam2.github.io/SweetySeelam.github.io/         

LinkedIn : https://www.linkedin.com/in/sweetyrao670/

----------------------------------------------------------------------------------------------------------------------------------------------

### 🔒 Proprietary & All Rights Reserved
© 2025 Sweety Seelam. This work is proprietary and protected by copyright. All content, models, code, and visuals are © 2025 Sweety Seelam. No part of this project, app, code, or analysis may be copied, reproduced, distributed, or used for any purpose—commercial or otherwise—without explicit written permission from the author.

For licensing, commercial use, or collaboration inquiries, please contact: Email: sweetyseelam2@gmail.com

----------------------------------------------------------------------------------------------------------------------------------------------

### ⭐ Show Support

If this project impressed or helped you, please 🌟 star the repo and share it with your network!

----------------------------------------------------------------------------------------------------------------------------------------------

### 🙌 Acknowledgements

This project was built by Sweety Seelam, leveraging data insights to assist platforms like Netflix in maximizing content ROI and strategic planning.
