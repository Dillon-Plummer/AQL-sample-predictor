import base64
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import statsmodels.api as sm
import numpy as np

# Configure Matplotlib to use a non-interactive backend
plt.switch_backend('Agg')


def fig_to_base64(fig):
    """Converts a Matplotlib figure to a base64 encoded string."""
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def create_trace_plot(trace):
    """Generates a trace plot from a PyMC trace."""
    fig, axes = plt.subplots(
        len(trace.posterior.data_vars), 2, figsize=(12, 8))
    az.plot_trace(trace, axes=axes)
    fig.tight_layout()
    return fig_to_base64(fig)


def create_posterior_plot(trace):
    """Generates a posterior distribution plot."""
    fig = az.plot_posterior(trace, figsize=(12, 6)).get_figure()
    fig.tight_layout()
    return fig_to_base64(fig)


def create_qq_plot(regression_results):
    """Generates a Q-Q plot of regression residuals."""
    residuals = regression_results.resid
    fig = sm.qqplot(residuals, line='s', fit=True)
    plt.title("Q-Q Plot of Residuals")
    plt.grid(True)
    return fig_to_base64(fig)


def create_importance_plot(importances):
    """Generates a bar plot of feature importances."""
    fig, ax = plt.subplots(figsize=(8, 5))
    importances.sort_values().plot(kind='barh', ax=ax)
    ax.set_title('Feature Importance for AQL Prediction')
    ax.set_xlabel('Importance')
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    fig.tight_layout()
    return fig_to_base64(fig)
