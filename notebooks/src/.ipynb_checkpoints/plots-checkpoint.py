import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (PredictionErrorDisplay)

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