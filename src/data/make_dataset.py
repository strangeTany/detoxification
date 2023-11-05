import pandas as pd
from matplotlib import pyplot as plt
import os

dirname = os.path.dirname(__file__)
raw_data_path = os.path.join(dirname, "../../data/raw/raw.tsv")
output_data_path = os.path.join(dirname, '../../data/interim/preprocessed.tsv')

raw_data = pd.read_csv(raw_data_path, sep='\t')

### make refernce column toxic and translation no toxic
cond = raw_data['ref_tox'] < raw_data['trn_tox']

raw_data.loc[cond, ['reference', 'translation']] = raw_data.loc[cond, ['translation', 'reference']].values
raw_data.loc[cond, ['ref_tox', 'trn_tox']] = raw_data.loc[cond, ['trn_tox', 'ref_tox']].values

### remove data with low similarity 
raw_data = raw_data.loc[raw_data['similarity']>0.63]

### remove data where small difference between reference toxicity and translated toxicity
raw_data = raw_data.loc[raw_data['ref_tox']-raw_data['trn_tox']>0.85]

raw_data.to_csv(output_data_path, sep="\t")