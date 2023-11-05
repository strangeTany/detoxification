from simpletransformers.seq2seq import Seq2SeqModel
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import evaluate

dirname = os.path.dirname(__file__)
data_path = os.path.join(dirname, '../../data/interim/preprocessed.tsv')
results_path = os.path.join(dirname, '../../data/interim/predict.tsv')
models_path = os.path.join(dirname, '../../models')

df = pd.read_csv(data_path, sep='\t')

df = df.rename(columns={"reference": "input_text", "translation": "target_text"})
train, test = train_test_split(df, test_size=0.2)
test = pd.DataFrame(test, columns = ['input_text','target_text', 'ref_tox', 'trn_tox']).sample(1000)
input = test['input_text']
target = test['target_text']

model = Seq2SeqModel(
    encoder_decoder_type="bart", 
    encoder_decoder_name=models_path,
    
)

predict = model.predict(input.tolist())

df = test.loc[input.index]
df["predict"] = predict
df.to_csv(results_path, sep="\t")


