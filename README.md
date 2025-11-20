OPHRP: Generalizability and Explainable Framework for AI-Driven Influenza Forecasting
This repository contains the supplementary materials, code structure, and data pipelines used for the study: "Generalizability and Explainable Framework for AI-Driven Influenza Forecasting using Multi-National Surveillance Data: Observational Retrospective Cohort Study."

1. Data Structure and Preparation

The model utilizes three primary time-series data streams: Epidemiological, Meteorological, and Digital (Google Search Trends).

Data Aggregation: All data is structured into a single unified time-series dataset with 305 weekly rows (Jan 2020 - Nov 2025) for each country (ROK, US, JP).

Feature Columns: Key columns include ILI_ROK, Hosp_US, ILI_JP (outcomes) and Temp_ROK, Humidity_US, Search_JP (covariates).


2. Modeling and Analysis Pipeline (Python/TensorFlow)
The core modeling and analysis process involves three sequential Python scripts:

File Name,Purpose,Description

1_preprocess.py,Data Cleaning and Preprocessing,"Handles missing value imputation (linear interpolation), Z-score normalization, and first-order differencing for stationarity check (ADF test). (Ref: Manuscript Materials and Methods)"

2_model_train.py,Model Training and LOCO-CV,"Defines and trains the SARIMA, LSTM, GRU, and Transformer architectures. Executes the Leave-One-Country-Out Cross-Validation (LOCO-CV) loop to assess generalization performance."

3_xai_analyze.py,Explainability and Bias Analysis,Calculates SHAP values for feature contribution and determines the Disparate Impact Ratio (DIR) to analyze potential bias in the national feature sets.


3. Environment and Dependencies
The following key Python packages are required to reproduce the results (Python 3.8+ recommended):

pandas, numpy (Data handling)

statsmodels (SARIMA and ADF test)

tensorflow/keras (Deep learning model construction)

shap (SHapley Additive exPlanations analysis)

scikit-learn (Performance metric calculation)


4. Contributing & License

We welcome contributions to enhance the reproducibility of the study. All code is released under a MIT license.

Contact: Chuelwon Lee   Date: November 2025