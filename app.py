from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# --- VEHICLE MODELS ---
vehicle_df = pd.read_csv('IEEE_DRDO_data.csv')

def preprocess_vehicle(df):
    data = df.copy()
    data['Purchase_Year'] = pd.to_datetime(data['Purchase Timestamp']).dt.year
    data['Purchase_Month'] = pd.to_datetime(data['Purchase Timestamp']).dt.month
    data['Vehicle_ID_Numeric'] = data['Vehicle ID'].str.extract('(\d+)').astype(int)
    data['Working_Condition'] = np.where(
        (data['Performance Metrics'] > 75) & (data['Number of Failures'] <= 3),
        1, 0
    )
    return data

vehicle_data = preprocess_vehicle(vehicle_df)
vehicle_features = ['Vehicle_ID_Numeric', 'Purchase_Year', 'Purchase_Month', 'Usage Hours', 'Maintenance Logs', 'Number of Failures']
X_vehicle = vehicle_data[vehicle_features]
y_vehicle = vehicle_data['Working_Condition']
X_train_vehicle, X_test_vehicle, y_train_vehicle, y_test_vehicle = train_test_split(X_vehicle, y_vehicle, test_size=0.5, random_state=15)
scaler_vehicle = StandardScaler()
X_train_vehicle_scaled = scaler_vehicle.fit_transform(X_train_vehicle)
X_test_vehicle_scaled = scaler_vehicle.transform(X_test_vehicle)

# Train models
vehicle_svm = SVC(probability=True, random_state=42)
vehicle_svm.fit(X_train_vehicle_scaled, y_train_vehicle)
vehicle_rf = RandomForestClassifier(n_estimators=100, random_state=42)
vehicle_rf.fit(X_train_vehicle_scaled, y_train_vehicle)
vehicle_dt = DecisionTreeClassifier(random_state=42)
vehicle_dt.fit(X_train_vehicle_scaled, y_train_vehicle)

# Evaluate models (for frontend display)
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='macro'),
        'recall': recall_score(y_test, y_pred, average='macro'),
        'f1': f1_score(y_test, y_pred, average='macro')
    }

vehicle_svm_metrics = evaluate_model(vehicle_svm, X_test_vehicle_scaled, y_test_vehicle)
vehicle_rf_metrics = evaluate_model(vehicle_rf, X_test_vehicle_scaled, y_test_vehicle)
vehicle_dt_metrics = evaluate_model(vehicle_dt, X_test_vehicle_scaled, y_test_vehicle)

# --- AIRCRAFT MODELS ---
aircraft_df = pd.read_csv('jets_data_updated.csv')
aircraft_df['Condition_Category'] = aircraft_df['Performance Metrics'].apply(
    lambda x: "Good condition" if x >= 90 else ("Working but needs attention" if x >= 80 else ("Work is required" if x >= 70 else "Retire"))
)
le_aircraft = LabelEncoder()
aircraft_df['Condition_Code'] = le_aircraft.fit_transform(aircraft_df['Condition_Category'])
aircraft_features = ['Performance Metrics', 'Usage Hours', 'Number of Failures', 'Maintenance Logs']
X_aircraft = aircraft_df[aircraft_features]
y_aircraft = aircraft_df['Condition_Code']
X_train_aircraft, X_test_aircraft, y_train_aircraft, y_test_aircraft = train_test_split(X_aircraft, y_aircraft, test_size=0.3, random_state=42)
scaler_aircraft = StandardScaler()
X_train_aircraft_scaled = scaler_aircraft.fit_transform(X_train_aircraft)
X_test_aircraft_scaled = scaler_aircraft.transform(X_test_aircraft)

aircraft_svm = SVC(probability=True, random_state=42)
aircraft_svm.fit(X_train_aircraft_scaled, y_train_aircraft)
aircraft_rf = RandomForestClassifier(n_estimators=100, random_state=42)
aircraft_rf.fit(X_train_aircraft_scaled, y_train_aircraft)
aircraft_dt = DecisionTreeClassifier(random_state=42)
aircraft_dt.fit(X_train_aircraft_scaled, y_train_aircraft)

aircraft_svm_metrics = evaluate_model(aircraft_svm, X_test_aircraft_scaled, y_test_aircraft)
aircraft_rf_metrics = evaluate_model(aircraft_rf, X_test_aircraft_scaled, y_test_aircraft)
aircraft_dt_metrics = evaluate_model(aircraft_dt, X_test_aircraft_scaled, y_test_aircraft)

# --- WEAPON MODELS ---
weapon_df = pd.read_csv('guns_data_with_environment.csv')
def categorize_performance(value):
    if value >= 85:
        return "Good Condition"
    elif value >= 70:
        return "Need Attention"
    else:
        return "Not Useable"
weapon_df['Performance Condition'] = weapon_df['Performance Metrics'].apply(categorize_performance)
le_weapon = LabelEncoder()
weapon_df['Condition_Code'] = le_weapon.fit_transform(weapon_df['Performance Condition'])
weapon_features = ['Usage Hours', 'Performance Metrics']
X_weapon = weapon_df[weapon_features]
y_weapon = weapon_df['Condition_Code']
X_train_weapon, X_test_weapon, y_train_weapon, y_test_weapon = train_test_split(X_weapon, y_weapon, test_size=0.4, random_state=42)
scaler_weapon = StandardScaler()
X_train_weapon_scaled = scaler_weapon.fit_transform(X_train_weapon)
X_test_weapon_scaled = scaler_weapon.transform(X_test_weapon)

weapon_svm = SVC(probability=True, random_state=42)
weapon_svm.fit(X_train_weapon_scaled, y_train_weapon)
weapon_rf = RandomForestClassifier(n_estimators=100, random_state=42)
weapon_rf.fit(X_train_weapon_scaled, y_train_weapon)
weapon_dt = DecisionTreeClassifier(random_state=42)
weapon_dt.fit(X_train_weapon_scaled, y_train_weapon)

weapon_svm_metrics = evaluate_model(weapon_svm, X_test_weapon_scaled, y_test_weapon)
weapon_rf_metrics = evaluate_model(weapon_rf, X_test_weapon_scaled, y_test_weapon)
weapon_dt_metrics = evaluate_model(weapon_dt, X_test_weapon_scaled, y_test_weapon)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/input/<eq_type>')
def input_page(eq_type):
    if eq_type.lower() == 'vehicle':
        metrics = {
            'svm': vehicle_svm_metrics,
            'rf': vehicle_rf_metrics,
            'dt': vehicle_dt_metrics
        }
    elif eq_type.lower() == 'aircraft':
        metrics = {
            'svm': aircraft_svm_metrics,
            'rf': aircraft_rf_metrics,
            'dt': aircraft_dt_metrics
        }
    elif eq_type.lower() == 'weapon':
        metrics = {
            'svm': weapon_svm_metrics,
            'rf': weapon_rf_metrics,
            'dt': weapon_dt_metrics
        }
    else:
        metrics = None
    return render_template('input.html', eq_type=eq_type.capitalize(), metrics=metrics)

@app.route('/predict', methods=['POST'])
def predict():
    eq_type = request.form['eq_type'].lower()
    eq_id = request.form['eq_id']
    model_type = request.form['model_type']

    if eq_type == 'vehicle':
        vehicle = vehicle_data[vehicle_data['Vehicle ID'] == eq_id]
        if vehicle.empty:
            return f"Vehicle ID {eq_id} not found."
        X = vehicle[vehicle_features]
        X_scaled = scaler_vehicle.transform(X)
        if model_type == 'svm':
            pred = vehicle_svm.predict(X_scaled)[0]
        elif model_type == 'rf':
            pred = vehicle_rf.predict(X_scaled)[0]
        else:
            pred = vehicle_dt.predict(X_scaled)[0]
        pred_label = 'Usable' if pred == 1 else 'Not Usable'
        details = {
            'Type': vehicle['Vehicle Type'].values[0],
            'Area': vehicle['Area'].values[0],
            'Battalion': vehicle['Battalion'].values[0],
            'Temperature': vehicle['Temperature'].values[0],
            'Weather Conditions': vehicle['Weather Conditions'].values[0],
            'Usage Hours': vehicle['Usage Hours'].values[0],
            'Number of Failures': vehicle['Number of Failures'].values[0],
            'Performance Metrics': vehicle['Performance Metrics'].values[0]
        }
    elif eq_type == 'aircraft':
        aircraft = aircraft_df[aircraft_df['Aircraft ID'] == eq_id]
        if aircraft.empty:
            return f"Aircraft ID {eq_id} not found."
        X = aircraft[aircraft_features]
        X_scaled = scaler_aircraft.transform(X)
        if model_type == 'svm':
            pred = aircraft_svm.predict(X_scaled)[0]
        elif model_type == 'rf':
            pred = aircraft_rf.predict(X_scaled)[0]
        else:
            pred = aircraft_dt.predict(X_scaled)[0]
        pred_label = le_aircraft.inverse_transform([pred])[0]
        details = {
            'Type': aircraft['Aircraft Type'].values[0],
            'Area': aircraft['Area'].values[0],
            'Battalion': aircraft['Battalion'].values[0],
            'Temperature': aircraft['Temperature'].values[0],
            'Weather Conditions': aircraft['Weather'].values[0],
            'Usage Hours': aircraft['Usage Hours'].values[0],
            'Number of Failures': aircraft['Number of Failures'].values[0],
            'Performance Metrics': aircraft['Performance Metrics'].values[0]
        }
    elif eq_type == 'weapon':
        weapon = weapon_df[weapon_df['Weapon ID'] == eq_id]
        if weapon.empty:
            return f"Weapon ID {eq_id} not found."
        X = weapon[weapon_features]
        X_scaled = scaler_weapon.transform(X)
        if model_type == 'svm':
            pred = weapon_svm.predict(X_scaled)[0]
        elif model_type == 'rf':
            pred = weapon_rf.predict(X_scaled)[0]
        else:
            pred = weapon_dt.predict(X_scaled)[0]
        pred_label = le_weapon.inverse_transform([pred])[0]
        details = {
            'Type': weapon['Weapon Type'].values[0],
            'Area': weapon['Area'].values[0],
            'Battalion': weapon['Battalion'].values[0],
            'Temperature': weapon['Temperature'].values[0],
            'Weather Conditions': weapon['Weather Conditions'].values[0],
            'Usage Hours': weapon['Usage Hours'].values[0],
            'Number of Failures': weapon['Number of Failures'].values[0],
            'Performance Metrics': weapon['Performance Metrics'].values[0]
        }
    else:
        return "Invalid equipment type."

    result = {
        'id': eq_id,
        'model_used': model_type.upper(),
        'prediction': pred_label,
        'details': details
    }
    return render_template('result.html', result=result, eq_type=eq_type.capitalize())

if __name__ == '__main__':
    app.run(debug=True)

