import pandas as pd
import numpy as np
import pymc as pm
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import shapiro, hypergeom
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import arviz as az

# --- Model Implementations ---


def run_bayesian_model(df: pd.DataFrame, prior_mean: float, prior_sigma: float):
    """
    Performs a Bayesian linear regression using PyMC.

    Args:
        df (pd.DataFrame): Input data with features and target.
        prior_mean (float): The mean for the coefficient priors.
        prior_sigma (float): The standard deviation for the coefficient priors.

    Returns:
        A tuple containing the PyMC trace and the summary DataFrame.
    """
    X = df[['lot_size', 'supplier_rating', 'material_density']].values
    y = df['defect_rate'].values

    # Standardize the features for better model performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    with pm.Model() as bayesian_regression:
        # Priors for the model parameters
        intercept = pm.Normal('Intercept', mu=0, sigma=1)
        # Use user-defined priors for coefficients
        betas = pm.Normal('betas', mu=prior_mean,
                          sigma=prior_sigma, shape=X_scaled.shape[1])

        # Prior for the error term
        sigma = pm.HalfNormal('sigma', sigma=1)

        # Linear model
        mu = intercept + pm.math.dot(X_scaled, betas)

        # Likelihood of the data
        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

        # Run the sampler
        trace = pm.sample(1000, tune=1000, cores=1)

    # Get summary statistics
    summary = az.summary(trace, var_names=['Intercept', 'betas', 'sigma'])

    return trace, summary


def run_linear_regression(df: pd.DataFrame):
    """
    Performs a Multiple Linear Regression using statsmodels for detailed diagnostics.

    Args:
        df (pd.DataFrame): Input data.

    Returns:
        A tuple containing the regression results object and test statistics.
    """
    X = df[['lot_size', 'supplier_rating', 'material_density']]
    y = df['defect_rate']
    X = sm.add_constant(X)  # Add a constant for the intercept

    model = sm.OLS(y, X)
    results = model.fit()

    # Perform residual analysis
    residuals = results.resid
    # Shapiro-Wilk test for normality of residuals
    shapiro_stat, shapiro_p = shapiro(residuals)

    # Breusch-Pagan test for homoscedasticity
    bp_test = het_breuschpagan(residuals, results.model.exog)
    bp_stat, bp_p = bp_test[0], bp_test[1]

    return results, shapiro_stat, shapiro_p, bp_stat, bp_p


def calculate_risk_score(df: pd.DataFrame, coeffs: pd.Series) -> float:
    """
    Calculates a risk score using a simplified hypergeometric approach.
    This is a conceptual implementation. A real-world model would be more complex.

    Args:
        df (pd.DataFrame): The input data.
        coeffs (pd.Series): The coefficients from the linear regression.

    Returns:
        A calculated risk score.
    """
    # Predict defect rate using regression coefficients
    X = df[['lot_size', 'supplier_rating', 'material_density']]
    X = sm.add_constant(X, has_constant='add')
    predicted_defect_rate = np.dot(X, coeffs)

    avg_predicted_defects = np.mean(predicted_defect_rate * df['lot_size'])

    # Hypergeometric parameters (conceptual)
    # M: total population size (average lot size)
    # n: number of defects in population (predicted)
    # N: sample size (e.g., 10% of lot size)
    M = int(df['lot_size'].mean())
    n = max(1, int(avg_predicted_defects))
    N = int(M * 0.1)

    # Calculate the probability of finding 1 or more defects in the sample
    # P(X >= 1) = 1 - P(X = 0)
    prob_of_zero_defects = hypergeom.pmf(0, M, n, N)
    risk_score = (1 - prob_of_zero_defects) * 100

    return risk_score


def run_random_forest_classifier(df: pd.DataFrame, risk_score: float):
    """
    Uses a pre-trained Random Forest model to predict the AQL.
    For this demo, we train a dummy model on the fly. In a real application,
    this model would be loaded from a file.

    Args:
        df (pd.DataFrame): The input data, used here to build the feature vector.
        risk_score (float): The calculated risk score from the previous step.

    Returns:
        A tuple with the predicted AQL, feature importances, and CV score.
    """
    # 1. Generate synthetic training data
    np.random.seed(42)
    n_samples = 100
    X_train_df = pd.DataFrame({
        'risk_score': np.random.rand(n_samples) * 100,
        'avg_lot_size': np.random.randint(500, 5000, n_samples),
        'avg_supplier_rating': np.random.uniform(3.5, 5.0, n_samples)
    })

    # Create a target variable 'aql' based on risk score
    conditions = [
        (X_train_df['risk_score'] > 75),
        (X_train_df['risk_score'] > 50),
        (X_train_df['risk_score'] > 25)
    ]
    choices = ['0.65', '1.0', '1.5']
    y_train = np.select(conditions, choices, default='2.5')

    # 2. Train the classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_df, y_train)

    # 3. Prepare the input for prediction
    X_predict = pd.DataFrame({
        'risk_score': [risk_score],
        'avg_lot_size': [df['lot_size'].mean()],
        'avg_supplier_rating': [df['supplier_rating'].mean()]
    })

    # 4. Make a prediction
    predicted_aql = model.predict(X_predict)[0]

    # 5. Get supporting information
    feature_importances = pd.Series(
        model.feature_importances_, index=X_train_df.columns)
    cv_score = cross_val_score(model, X_train_df, y_train, cv=5).mean()

    return predicted_aql, feature_importances, cv_score
