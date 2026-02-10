import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
import seaborn as sns

from sklearn.metrics import (PredictionErrorDisplay)

from .models import RANDOM_STATE

sns.set_theme(palette='bright')

SCATTER_ALPHA = 0.2
PALETTE = 'icefire'

def plot_coefs(coefs_dataFrame, title='coefs'):
    return coefs_dataFrame.plot.barh(
        legend=False,
        xlabel = 'coefficients',
        title=title
    )

def plot_residuals(y_test, y_pred):
    fis, axs = plt.subplots(1, 3, figsize=(10,4))
    
    PredictionErrorDisplay.from_predictions(
        y_true=y_test, 
        y_pred=y_pred,
        ax=axs[0]
    )
    
    PredictionErrorDisplay.from_predictions(
        y_true=y_test, 
        y_pred=y_pred,
        kind='actual_vs_predicted',
        ax=axs[1]
    )
    
    residuals = y_test - y_pred
    sns.histplot(data=residuals, kde=True, ax=axs[2])
    
    plt.tight_layout()
    
    plt.show()
    
def plot_compare_models_metrics(df):
    fig, axs = plt.subplots(2, 2, figsize=(8,8), sharex=True)

    metrics = [
        'time_seconds',
        'test_r2',
        'test_neg_mean_absolute_error',
        'test_neg_root_mean_squared_error',
    ]

    rename_metrics = [
        'Tempo (s)',
        'R2',
        'MAE',
        'RMSE'
    ]

    for ax, metric, name in zip(axs.flatten(), metrics, rename_metrics):
        sns.boxplot(
            x='model',
            y=metric,
            data=df,
            ax=ax,
            showmeans=True    
        )
        ax.set_title(name)
        ax.tick_params(axis='x', rotation=90)

    plt.tight_layout()
    plt.show()
    
def plot_residuals_estimator(estimator, X, y, subsample=0.25, eng_formatter=True):
    fis, axs = plt.subplots(1, 3, figsize=(10,4))
    
    error_display_01 = PredictionErrorDisplay.from_estimator(
        estimator=estimator,
        X=X,
        y=y,
        kind='residual_vs_predicted',
        ax=axs[1],
        random_state=RANDOM_STATE,
        scatter_kwargs={'alpha': SCATTER_ALPHA},
        subsample=subsample
    )
    
    error_display_01 = PredictionErrorDisplay.from_estimator(
        estimator=estimator,
        X=X,
        y=y,
        kind='actual_vs_predicted',
        ax=axs[2],
        random_state=RANDOM_STATE,
        scatter_kwargs={'alpha': SCATTER_ALPHA},
        subsample=subsample
    )
    
    residuals = error_display_01.y_true - error_display_01.y_pred
    
    sns.histplot(data=residuals, kde=True, ax=axs[0])
    
    if eng_formatter:
        for ax in axs:
            ax.yaxis.set_major_formatter(EngFormatter())
            ax.xaxis.set_major_formatter(EngFormatter())
            
    plt.tight_layout()
    
    plt.show()