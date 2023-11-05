## Ivshina Tatiana

t.ivshina@innopolis.university <br>
Group BS20-DS-01

## Requirements

For data preparation run `src/data/make_dataset.py`. It is required that `raw.tsv` file with initial data is in `data/raw` folder. <br>
To train model run `src/models/train_model.py`.
To get detoxified text from test part of dataset run `src/models/predict_model.py`. Make sure that either data preparation step was runned or file `data/interim/preprocessed.tsv` exists. <br>
For visualization you need all previous steps done and run `src/visualization/visualize.py`

## Acknoledgements:

Training and prediction was done using <a href="https://github.com/ThilinaRajapakse/simpletransformers/tree/master" title="simpletransformers">simpletransformers</a> open source library and a pretrained <a href="https://huggingface.co/eugenesiow/bart-paraphrase/blob/main/README.md?code=true" title="bart paraphraser">bart paraphraser</a> based on it.
<br>
Evaluation was done using <a href="https://huggingface.co/spaces/evaluate-measurement/toxicity" title="toxicity metrics">toxicity metric</a>.
