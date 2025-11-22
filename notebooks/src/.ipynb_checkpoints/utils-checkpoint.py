import pandas as pd

def dataframe_coefs(coefs, index_label):
    return pd.DataFrame(
        data = coefs,
        index = index_label,
        columns = ['coefficients'],
    ).sort_values(by='coefficients')