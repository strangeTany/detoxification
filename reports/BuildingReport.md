# Building Reprot

Firstly I want to evaluate initial data quality and improve it. Also I think I need to reduce dataset size due to resource shortage so I think I can make dataset more quality with cutting off bad samples.

Secondlly, for the modeling part I think it is valid to try similar language models with ones in the original paper. It can be either trained model from scratch with detoxification paraphresing or some pre trained parphrasing model with fine-tuning for detoxification.

Finally, for evoluation I need to use 3 metrics: similarity, toxicity and human metric by visual inspection. I think human metric is required since it is needed for sentences to preserve humanity in predicted sentences.
