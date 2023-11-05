from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs
from sklearn.model_selection import train_test_split
import pandas as pd
import os

### Read data and train/test split
dirname = os.path.dirname(__file__)
data_path = os.path.join(dirname, '../../data/interim/preprocessed.tsv')
models_path = os.path.join(dirname, '../../models')

df = pd.read_csv(data_path, sep='\t')

df = df.rename(columns={"reference": "input_text", "translation": "target_text"})
train, test = train_test_split(df, test_size=0.2)
train = pd.DataFrame(train, columns = ['input_text','target_text'])
test = pd.DataFrame(test, columns = ['input_text','target_text'])


###initializing model arguments
model_args = Seq2SeqArgs()
model_args.eval_batch_size = 64
model_args.evaluate_generated_text = True
model_args.evaluate_during_training = True
model_args.evaluate_during_training_steps = 2500
model_args.evaluate_during_training_verbose = True
model_args.fp16 = False
model_args.learning_rate = 5e-5
model_args.max_seq_length = 128
model_args.num_train_epochs = 2
model_args.overwrite_output_dir = True
model_args.reprocess_input_data = True
model_args.save_eval_checkpoints = False
model_args.save_steps = -1
model_args.train_batch_size = 8
model_args.use_multiprocessing = False

model_args.do_sample = True
model_args.num_beams = 1
model_args.num_return_sequences = 1
model_args.max_length = 128
model_args.top_k = 50
model_args.top_p = 0.95
model_args.output_dir = models_path

### model init

model = Seq2SeqModel(
    encoder_decoder_type="bart",
    encoder_decoder_name="eugenesiow/bart-paraphrase",
    args=model_args,
    use_cuda=False,
    use_mps_device = True
)

### train
model.train_model(train, eval_data=test)