import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import os

import evaluate


dirname = os.path.dirname(__file__)
raw_data_path = os.path.join(dirname, "../../data/raw/raw.tsv")
pred_path = os.path.join(dirname, "../../data/interim/predict.tsv")
picture_path = os.path.join(dirname, '../../reports/figures')

raw_data = pd.read_csv(raw_data_path, sep='\t')

### make reference a toxic sentences and translated detoxed sentences
cond = raw_data['ref_tox'] < raw_data['trn_tox']

raw_data.loc[cond, ['reference', 'translation']] = raw_data.loc[cond, ['translation', 'reference']].values
raw_data.loc[cond, ['ref_tox', 'trn_tox']] = raw_data.loc[cond, ['trn_tox', 'ref_tox']].values
raw_data.describe()

### similarity plot
plot = sns.displot(x=raw_data['similarity'], alpha=0.5, linewidth=0.3)
bbox = dict(facecolor='green', alpha=0.3, pad=0.05, edgecolor='none')
similarity_bb = raw_data["similarity"].quantile(0.1)
plt.axvline(similarity_bb, 0, 1, color="black", linewidth=0.75)
plt.text(similarity_bb, 0, similarity_bb.round(5), fontsize=8, bbox=bbox)

plot.savefig(os.path.join(picture_path, ('similarity' + ".png")))
plt.clf()

### difference between ref_tox and trn_tox
plot = sns.displot(x=raw_data['ref_tox']-raw_data['trn_tox'], alpha=0.5, linewidth=0.3)
bbox = dict(facecolor='green', alpha=0.3, pad=0.05, edgecolor='none')
similarity_bb = (raw_data['ref_tox']-raw_data['trn_tox']).quantile(0.2)
plt.axvline(similarity_bb, 0, 1, color="black", linewidth=0.75)
plt.text(similarity_bb, 0, similarity_bb.round(5), fontsize=8, bbox=bbox)

plot.savefig(os.path.join(picture_path, ('toxicity_difference' + ".png")))
plt.clf()


df = pd.read_csv(pred_path, sep='\t')

toxicity = evaluate.load("toxicity", module_type="measurement")
results = toxicity.compute(predictions=df['predict'].apply(lambda x: x.lower()).to_list())
ref_tox = toxicity.compute(predictions=df['input_text'].apply(lambda x: x.lower()).to_list())
trn_tox = toxicity.compute(predictions=df['target_text'].apply(lambda x: x.lower()).to_list())

df['predict_tox'] = pd.Series(results['toxicity'])
ref_tox, trn_tox = ref_tox["toxicity"], trn_tox["toxicity"]

df_ref = pd.DataFrame({"tox": pd.Series(ref_tox),
                       "label": pd.Series(["ref"] * df["ref_tox"].shape[0])})
df_trn = pd.DataFrame({"tox": pd.Series(trn_tox),
                       "label": pd.Series(["trn"] * df["trn_tox"].shape[0])})
df_predict = pd.DataFrame({"tox": df["predict_tox"],
                       "label": pd.Series(["predict"] * df["predict_tox"].shape[0])})
df_scores = pd.concat([df_ref, df_trn, df_predict]).reset_index(drop=True)

plot = sns.displot(data=df_scores, x="tox", hue="label")
plot.savefig(os.path.join(picture_path, ('toxicity_prediction' + ".png")))
plt.clf()

sns.displot(x=pd.Series(trn_tox)-df['predict_tox'], alpha=0.5, linewidth=0.3)
plot.savefig(os.path.join(picture_path, ('prediction_target_difference' + ".png")))
plt.clf()