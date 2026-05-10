from flask import Flask, render_template, jsonify, request, redirect, url_for, session, flash
from flask_mail import Mail, Message
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
from functools import wraps
from preprocessing import load_data, preprocess_data
from model import build_lstm_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import database

app = Flask(__name__)
app.secret_key = 'energystream_ai_secret_key' # In production, use a secure environment variable

# --- Flask-Mail Configuration ---
# WARNING: Update these with your real SMTP credentials
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'your-email@gmail.com' 
app.config['MAIL_PASSWORD'] = 'your-app-password' # Use a Gmail App Password
app.config['MAIL_DEFAULT_SENDER'] = 'your-email@gmail.com'

mail = Mail(app)
CORS(app)

# Global variables for convenience and caching
DATA_FILE = 'household_power_consumption.txt'
scaler = None
cols = None
df_resampled = None
cached_X = None
cached_y = None

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def get_preprocessed_info():
    global scaler, cols, df_resampled, cached_X, cached_y
    if df_resampled is None:
        print("Initial data load and preprocess...")
        df = load_data(DATA_FILE)
        # Note: preprocess_data now handles resampling internally if needed
        cached_X, cached_y, scaler, cols = preprocess_data(df)
        df_resampled = df.resample('h').mean().fillna(df.mean())
    return df_resampled, scaler, cols, cached_X, cached_y

def invert_scaling(data, scaler, num_features, target_idx):
    dummy = np.zeros((len(data), num_features))
    dummy[:, target_idx] = data.flatten()
    inv = scaler.inverse_transform(dummy)
    return inv[:, target_idx]

# --- Auth Routes ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = database.get_user_by_username(username)
        
        if user and database.verify_password(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash('Successfully logged in!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password.', 'error')
            
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if database.get_user_by_username(username):
            flash('Username already exists.', 'error')
        else:
            if database.create_user(username, email, password):
                flash('Account created! Please log in.', 'success')
                return redirect(url_for('login'))
            else:
                flash('An error occurred. Please try again.', 'error')
                
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        username = request.form.get('username')
        user = database.get_user_by_username(username)
        
        if user and user['email'] == email:
            try:
                msg = Message(
                    subject="Password Recovery - EnergyStream AI",
                    recipients=[email],
                    body=f"Hello {username},\n\nYou requested password recovery. Your current password hash is: {user['password']}\n\nPlease contact your administrator to reset it."
                )
                mail.send(msg)
                
                # Still log locally for debugging
                recovery_msg = f"PASSWORD RECOVERY for {username} ({email}): Sent to inbox."
                print(f"\n🚀 [EMAIL SENT] to {email}")
                
                with open('recovery_logs.txt', 'a') as f:
                    f.write(f"[{pd.Timestamp.now()}] {recovery_msg}\n")
                    
                flash(f'Recovery email sent to {email}! Please check your inbox.', 'success')
            except Exception as e:
                print(f"❌ [EMAIL ERROR]: {str(e)}")
                flash(f'Failed to send email. Error: {str(e)}', 'error')
        else:
            flash('Username and Email do not match our records.', 'error')
    return render_template('forgot_password.html')

# --- Core Routes ---
@app.route('/')
@login_required
def home():
    return render_template('home.html')

@app.route('/forecast')
@login_required
def forecast_page():
    return render_template('forecast.html')

@app.route('/analysis')
@login_required
def analysis_page():
    return render_template('analysis.html')

@app.route('/about')
@login_required
def about_page():
    db_abs_path = os.path.abspath(database.DB_PATH)
    return render_template('about.html', db_path=db_abs_path)

@app.route('/api/stats', methods=['GET'])
@login_required
def get_stats():
    df, _, _, _, _ = get_preprocessed_info()
    stats = {
        "total_records": len(df),
        "columns": list(df.columns),
        "mean_active_power": float(df['Global_active_power'].mean()),
        "max_active_power": float(df['Global_active_power'].max()),
        "last_updated": df.index[-1].strftime('%Y-%m-%d %H:%M:%S')
    }
    return jsonify(stats)

@app.route('/api/predict')
@login_required
def predict_all():
    df, scaler, cols, X, y = get_preprocessed_info()
    
    if X is None or len(X) == 0:
        return jsonify({"error": "No data available"}), 400

    # Split for comparison
    test_size = 200
    X_test = X[-test_size:]
    y_test = y[-test_size:]
    
    # Use a smaller subset for quick "online" training if necessary, 
    # or just use the model as is. For now, keeping it quick.
    train_subset_size = 500
    X_train = X[-train_subset_size-test_size:-test_size]
    y_train = y[-train_subset_size-test_size:-test_size]

    input_shape = (X.shape[1], X.shape[2])
    lstm_model = build_lstm_model(input_shape)
    
    # Extremely limited training for demo speed, or skip if pre-trained (not implemented yet)
    lstm_model.fit(X_train, y_train, epochs=1, batch_size=72, verbose=0)
    y_pred_lstm = lstm_model.predict(X_test)

    # Traditional models
    X_trad_train = X_train.reshape(len(X_train), -1)
    X_trad_test = X_test.reshape(len(X_test), -1)
    y_trad_train = y_train.flatten()

    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rf.fit(X_trad_train, y_trad_train)
    y_pred_rf = rf.predict(X_trad_test)

    dt = DecisionTreeRegressor(random_state=42)
    dt.fit(X_trad_train, y_trad_train)
    y_pred_dt = dt.predict(X_trad_test)

    num_features = len(cols)
    target_idx = list(cols).index('Global_active_power')
    
    y_test_inv = invert_scaling(y_test, scaler, num_features, target_idx)
    y_pred_lstm_inv = invert_scaling(y_pred_lstm, scaler, num_features, target_idx)
    y_pred_rf_inv = invert_scaling(y_pred_rf.reshape(-1, 1), scaler, num_features, target_idx)
    y_pred_dt_inv = invert_scaling(y_pred_dt.reshape(-1, 1), scaler, num_features, target_idx)

    response = {
        "labels": [df.index[-test_size + i].strftime('%H:%M') for i in range(test_size)],
        "actual": y_test_inv.tolist(),
        "lstm": y_pred_lstm_inv.tolist(),
        "rf": y_pred_rf_inv.tolist(),
        "dt": y_pred_dt_inv.tolist(),
        "metrics": {
            "lstm_rmse": float(np.sqrt(mean_squared_error(y_test_inv, y_pred_lstm_inv))),
            "rf_rmse": float(np.sqrt(mean_squared_error(y_test_inv, y_pred_rf_inv))),
            "dt_rmse": float(np.sqrt(mean_squared_error(y_test_inv, y_pred_dt_inv)))
        }
    }
    return jsonify(response)

@app.route('/api/predict_manual', methods=['POST'])
@login_required
def predict_manual():
    try:
        data = request.json
        v = float(data.get('Voltage', 238.5))
        intensity = float(data.get('Global_intensity', 5.2))
        reactive = float(data.get('Global_reactive_power', 0.124))
        
        df, scaler, cols, X, y = get_preprocessed_info()
        target_col = 'Global_active_power'
        target_idx = list(cols).index(target_col)
        
        full_df = df[cols].copy()
        scaled_full = scaler.transform(full_df)
        last_23 = scaled_full[-23:]
        
        dummy_row = np.zeros((1, len(cols)))
        for i, col in enumerate(cols):
            if col == 'Voltage': dummy_row[0, i] = v
            elif col == 'Global_intensity': dummy_row[0, i] = intensity
            elif col == 'Global_reactive_power': dummy_row[0, i] = reactive
            elif col == target_col: dummy_row[0, i] = 0
            else: dummy_row[0, i] = df[col].mean()
            
        scaled_user_row = scaler.transform(dummy_row)[0]
        new_window = np.vstack([last_23, scaled_user_row]).reshape(1, 24, -1)
        
        input_shape = (new_window.shape[1], new_window.shape[2])
        lstm_model = build_lstm_model(input_shape)
        pred_scaled = lstm_model.predict(new_window)
        pred_val = invert_scaling(pred_scaled, scaler, len(cols), target_idx)
        
        return jsonify({
            "success": True,
            "prediction": float(pred_val[0]),
            "unit": "kW"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

if __name__ == '__main__':
    if not os.path.exists(DATA_FILE):
        print(f"CRITICAL: {DATA_FILE} not found!")
    else:
        app.run(debug=True, port=5000)
