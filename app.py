import os
import uuid
import pandas as pd
from flask import Flask, request, jsonify, render_template, session
from werkzeug.utils import secure_filename

# Import our custom modules
import models
import utils

# --- App Configuration ---
app = Flask(__name__)
# A secret key is required for using sessions. Change this in a real deployment.
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'a-default-fallback-secret-key')
# Use a temporary directory that is standard in production environments
app.config['UPLOAD_FOLDER'] = '/tmp'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Helper Function ---
def get_df_from_session():
    """Safely retrieves the DataFrame from the filepath stored in the session."""
    filepath = session.get('filepath')
    if not filepath or not os.path.exists(filepath):
        return None
    try:
        return pd.read_csv(filepath)
    except Exception:
        return None

# --- Main Route ---
@app.route('/')
def index():
    """Renders the main application page."""
    # Clear session on new visit to start fresh
    filepath = session.pop('filepath', None)
    if filepath and os.path.exists(filepath):
        os.remove(filepath) # Clean up old files
    return render_template('index.html')

# --- Data Handling Routes ---
@app.route('/upload_data', methods=['POST'])
def upload_data():
    """
    Handles data loading. Instead of storing the data in the session cookie,
    it saves the file to a temporary location and stores only the file path
    in the session. This avoids the 4KB cookie size limit.
    """
    source = request.form.get('source')
    
    # Generate a unique filename to prevent conflicts
    unique_filename = f"{uuid.uuid4()}.csv"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    
    try:
        if source == 'demo':
            df = pd.DataFrame({
                'lot_size': [1000, 1500, 800, 2000, 1200, 1800, 950, 1300],
                'supplier_rating': [4.5, 4.2, 4.8, 3.9, 4.6, 4.1, 4.7, 4.4],
                'material_density': [2.7, 2.75, 2.68, 2.8, 2.71, 2.77, 2.69, 2.72],
                'defect_rate': [0.02, 0.03, 0.01, 0.05, 0.025, 0.04, 0.015, 0.028]
            })
            df.to_csv(filepath, index=False)
            message = f'Demo data loaded successfully ({len(df)} records).'
            filename_for_display = 'Demo Data'
            
        elif source == 'upload' and 'csvFile' in request.files:
            file = request.files['csvFile']
            if not file or not file.filename.endswith('.csv'):
                return jsonify({'status': 'error', 'message': 'Invalid file type. Please upload a CSV.'}), 400
            
            file.save(filepath)
            df = pd.read_csv(filepath) # Read once to validate
            required_cols = {'lot_size', 'supplier_rating', 'material_density', 'defect_rate'}
            if not required_cols.issubset(df.columns):
                os.remove(filepath) # Clean up invalid file
                return jsonify({'status': 'error', 'message': f'CSV must contain columns: {required_cols}'}), 400
            
            message = f'File "{secure_filename(file.filename)}" loaded successfully ({len(df)} records).'
            filename_for_display = secure_filename(file.filename)
        else:
            return jsonify({'status': 'error', 'message': 'Invalid request.'}), 400

        # Store ONLY the filepath in the session
        session['filepath'] = filepath
        
        return jsonify({
            'status': 'success',
            'message': message,
            'filename': filename_for_display
        })
        
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath) # Clean up on error
        return jsonify({'status': 'error', 'message': f'Error processing data: {e}'}), 500

# --- Model Execution Routes ---

@app.route('/run_step_1', methods=['POST'])
def run_step_1():
    """Runs the Bayesian model."""
    df = get_df_from_session()
    if df is None:
        return jsonify({'status': 'error', 'message': 'Data not found or session expired. Please load data again.'}), 400
        
    data = request.get_json()
    prior_mean = float(data.get('prior_mean', 0))
    prior_sigma = float(data.get('prior_sigma', 1))
    
    try:
        trace, summary = models.run_bayesian_model(df, prior_mean, prior_sigma)
        session['posterior_summary'] = summary.to_json() # Summary is small, OK for session
        
        trace_plot_b64 = utils.create_trace_plot(trace)
        posterior_plot_b64 = utils.create_posterior_plot(trace)
        summary_html = summary.to_html(classes='table-auto w-full text-sm text-left')
        
        return jsonify({
            'status': 'success',
            'trace_plot': trace_plot_b64,
            'posterior_plot': posterior_plot_b64,
            'summary_html': summary_html
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Error in Bayesian model: {e}'}), 500

@app.route('/run_step_2', methods=['POST'])
def run_step_2():
    """Runs the Multiple Linear Regression model."""
    df = get_df_from_session()
    if df is None:
        return jsonify({'status': 'error', 'message': 'Data not found or session expired. Please load data again.'}), 400

    try:
        results, shapiro_stat, shapiro_p, bp_stat, bp_p = models.run_linear_regression(df)
        session['regression_coeffs'] = results.params.to_json() # Coeffs are small, OK for session

        qq_plot_b64 = utils.create_qq_plot(results)
        summary_html = results.summary().as_html()
        
        return jsonify({
            'status': 'success',
            'summary_html': summary_html,
            'qq_plot': qq_plot_b64,
            'shapiro_stat': f'{shapiro_stat:.4f}',
            'shapiro_p': f'{shapiro_p:.4f}',
            'bp_stat': f'{bp_stat:.4f}',
            'bp_p': f'{bp_p:.4f}'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Error in Linear Regression: {e}'}), 500

@app.route('/run_step_3', methods=['POST'])
def run_step_3():
    """Calculates the Hypergeometric risk score."""
    df = get_df_from_session()
    coeffs_json = session.get('regression_coeffs')
    if df is None or not coeffs_json:
        return jsonify({'status': 'error', 'message': 'Previous step results not found or session expired.'}), 400

    coeffs = pd.read_json(coeffs_json, typ='series')
    
    try:
        risk_score = models.calculate_risk_score(df, coeffs)
        session['risk_score'] = risk_score
        
        return jsonify({
            'status': 'success',
            'risk_score': f'{risk_score:.2f}'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Error calculating risk score: {e}'}), 500

@app.route('/run_step_4', methods=['POST'])
def run_step_4():
    """Runs the Random Forest Classifier to get the final AQL."""
    df = get_df_from_session()
    risk_score = session.get('risk_score')
    if df is None or risk_score is None:
        return jsonify({'status': 'error', 'message': 'Risk score not found or session expired.'}), 400
    
    try:
        aql, feature_importances, cv_score = models.run_random_forest_classifier(df, risk_score)
        session['final_aql'] = aql

        importance_plot_b64 = utils.create_importance_plot(feature_importances)
        
        return jsonify({
            'status': 'success',
            'final_aql': aql,
            'cv_score': f'{cv_score:.3f}',
            'importance_plot': importance_plot_b64
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Error in Random Forest model: {e}'}), 500
