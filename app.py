import os
import pandas as pd
from flask import Flask, request, jsonify, render_template, session
from werkzeug.utils import secure_filename

# Import our custom modules
import models
import utils

# --- App Configuration ---
app = Flask(__name__)
# A secret key is required for using sessions
app.config['SECRET_KEY'] = 'your-super-secret-key-change-me'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Main Route ---
@app.route('/')
def index():
    """Renders the main application page."""
    # Clear session on new visit to start fresh
    session.clear()
    return render_template('index.html')

# --- Data Handling Routes ---
@app.route('/upload_data', methods=['POST'])
def upload_data():
    """Handles both CSV file uploads and demo data loading."""
    source = request.form.get('source')
    
    if source == 'demo':
        # Use built-in demo data
        df = pd.DataFrame({
            'lot_size': [1000, 1500, 800, 2000, 1200, 1800, 950, 1300],
            'supplier_rating': [4.5, 4.2, 4.8, 3.9, 4.6, 4.1, 4.7, 4.4],
            'material_density': [2.7, 2.75, 2.68, 2.8, 2.71, 2.77, 2.69, 2.72],
            'defect_rate': [0.02, 0.03, 0.01, 0.05, 0.025, 0.04, 0.015, 0.028]
        })
        # Store dataframe as JSON in session
        session['df'] = df.to_json()
        return jsonify({
            'status': 'success',
            'message': f'Demo data loaded successfully ({len(df)} records).',
            'filename': 'Demo Data'
        })
        
    elif source == 'upload' and 'csvFile' in request.files:
        file = request.files['csvFile']
        if file and file.filename.endswith('.csv'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                df = pd.read_csv(filepath)
                # Basic validation: check for required columns
                required_cols = {'lot_size', 'supplier_rating', 'material_density', 'defect_rate'}
                if not required_cols.issubset(df.columns):
                    return jsonify({'status': 'error', 'message': f'CSV must contain columns: {required_cols}'}), 400
                
                # Store dataframe in session
                session['df'] = df.to_json()
                return jsonify({
                    'status': 'success',
                    'message': f'File "{filename}" loaded successfully ({len(df)} records).',
                    'filename': filename
                })
            except Exception as e:
                return jsonify({'status': 'error', 'message': f'Error processing CSV file: {e}'}), 500
        else:
            return jsonify({'status': 'error', 'message': 'Invalid file type. Please upload a CSV.'}), 400
    
    return jsonify({'status': 'error', 'message': 'Invalid request.'}), 400


# --- Model Execution Routes ---

@app.route('/run_step_1', methods=['POST'])
def run_step_1():
    """Runs the Bayesian model."""
    if 'df' not in session:
        return jsonify({'status': 'error', 'message': 'Data not found. Please load data first.'}), 400
        
    data = request.get_json()
    prior_mean = float(data.get('prior_mean', 0))
    prior_sigma = float(data.get('prior_sigma', 1))
    
    df = pd.read_json(session['df'])
    
    try:
        # Run the Bayesian model
        trace, summary = models.run_bayesian_model(df, prior_mean, prior_sigma)
        session['posterior_summary'] = summary.to_json()
        
        # Generate plots
        trace_plot_b64 = utils.create_trace_plot(trace)
        posterior_plot_b64 = utils.create_posterior_plot(trace)
        
        # Format summary for display
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
    if 'df' not in session:
        return jsonify({'status': 'error', 'message': 'Data not found.'}), 400

    df = pd.read_json(session['df'])
    
    try:
        results, shapiro_stat, shapiro_p, bp_stat, bp_p = models.run_linear_regression(df)
        session['regression_coeffs'] = results.params.to_json()

        # Generate QQ plot
        qq_plot_b64 = utils.create_qq_plot(results)

        # Format results for display
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
    if 'df' not in session or 'regression_coeffs' not in session:
        return jsonify({'status': 'error', 'message': 'Previous step results not found.'}), 400

    df = pd.read_json(session['df'])
    coeffs = pd.read_json(session['regression_coeffs'], typ='series')
    
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
    if 'risk_score' not in session or 'df' not in session:
        return jsonify({'status': 'error', 'message': 'Risk score not found.'}), 400

    risk_score = session['risk_score']
    df = pd.read_json(session['df'])
    
    try:
        aql, feature_importances, cv_score = models.run_random_forest_classifier(df, risk_score)
        session['final_aql'] = aql

        # Generate feature importance plot
        importance_plot_b64 = utils.create_importance_plot(feature_importances)
        
        return jsonify({
            'status': 'success',
            'final_aql': aql,
            'cv_score': f'{cv_score:.3f}',
            'importance_plot': importance_plot_b64
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Error in Random Forest model: {e}'}), 500
