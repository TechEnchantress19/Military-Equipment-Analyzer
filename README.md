# Military Equipment Analyzer – Predictive Maintenance in Defense

## Overview 
The Military Equipment Analyzer is a predictive maintenance system built using machine learning to assess the performance and readiness of mission-critical military assets such as Fighter Jets, Weapons, and Vehicles.
This project helps defense systems transition from reactive to proactive maintenance by using equipment data to predict performance and provide actionable insights on whether an asset should be maintained, monitored, or retired.

## Problem Statement
In modern defense, equipment failure during critical missions can be catastrophic. Regular manual checks are labor-intensive and inefficient. Our goal is to:
+ Analyze performance data from military equipment
+ Predict operational status using classification models
+ Display insights through a user-friendly web dashboard

## Machine Learning Models
We implemented and compared the following ML models:
+ `Random Forest Classifier` – Ensemble method that uses multiple decision trees.
+  `Support Vector Machine (SVM)` – Good for high-dimensional data with strong classification boundaries.

## Data Description
Each equipment dataset includes parameters such as:
+ Type (Weapon / Jet / Vehicle)
+ Battalion Area
+ Temperature & Weather Conditions
+ Usage Hours
+ Failure Incidents
+ Performance Metrics

## Datasets
+ jets_data_updated.csv
+ guns_data_with_environment.csv
+ IEEE_DRDO_data.csv

## Web Application

We built a Flask-based dashboard with a clean, responsive frontend.

## Key Features
+ Input `Equipment ID` to view current status
+ `Usage` vs `Performance` visualizations
+ `Temperature` & `Environment` analysis
+ Light/Dark Mode Toggle
+ Separate category folders for each `equipment type`

## How It Works (Workflow)
+ Data Ingestion – `Load .csv` files into pandas DataFrames
+ Preprocessing – Handle `nulls`, convert `categories`, scale `values`
+ Model Training – Train `Random Forest` and `SVM models`
+ Model Evaluation – `Accuracy`, `Confusion Matrix`, `Precision`, `Recall`
+ Flask Integration – Wrap model in `app.py` for prediction
+ Visualization – Render plots via `matplotlib/seaborn`

## Future Enhancements
+ `IoT Integration` – Real-time sensor input from military assets
+ `Live Dashboard` – Continuous status updates from units
+ `Federated Learning` – Train across branches without sharing sensitive data
+ `Multi-force Deployment` – Extend to Army, Navy, and Air Force

<h3>Tech Stack</h3>

`Python` ,`pandas`, `numpy`, `scikit-learn`, `Flask`, `HTML`, `CSS`, `JS`, `MySQL (optional for data storage)`, `Jupyter Notebook`, `VS Code`

## Installation & Setup
+ Install dependencies: `pip install -r requirements.txt`  
+ Run the Flask App: `python app.py`
+ Open in browser: `[Local Host ID]`

## Contributors
+ Mentor: [Dr. Tanupriya Choudhury](https://www.linkedin.com/in/tanupriya-choudhury-0811b926/)
+ [Sukalp Jhingran](https://www.linkedin.com/in/sukalp-jhingran-485162262/)
+ [Hardik Raj Kapoor](https://www.linkedin.com/in/hardik-raj-kapoor/)

## License
This project is intended for academic demonstration purposes only. For external use, please contact the maintainers.
Stay Mission-Ready. Predict. Prevent. Perform.
