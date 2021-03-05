import os, sys
from scipy.io import arff
import pandas as pd
directory = os.path.dirname(os.path.realpath(__file__))

data, meta = arff.loadarff(directory+'/dataset.arff')
df = pd.DataFrame(data)
results = df['Result'].to_numpy()
n_samples, n_attributes = df.shape

group_names = [
    'Address_bar_based',
    'Abnormal_based',
    'HTML_and_JavaScript_based',
    'Domain_based'
]
groups = {
    'Address_bar_based' : df.columns[0:12],
    'Abnormal_based' : df.columns[12:18],
    'HTML_and_JavaScript_based' : df.columns[18:23],
    'Domain_based' : df.columns[23:30]
    }

features = []
for group in group_names:
    features.append(df[groups[group]].to_numpy())

def generate_data(group,n_samples=None):
    assert group in groups
    features = groups[group]
    if n_samples is None:
        return df[features].to_numpy()
    else:
        assert n_samples <= df.shape[0]
        return df[features][0:n_samples].to_numpy()
    
