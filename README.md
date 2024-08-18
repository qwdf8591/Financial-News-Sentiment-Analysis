# Financial-News-Sentiment-Analysis

# roberta-base_auditor_sentiment

This model is based on [roberta-base](https://huggingface.co/roberta-base) trainning from [auditor_sentiment](https://huggingface.co/datasets/FinanceInc/auditor_sentiment?row=23) dataset.

It achieves the following results on the evaluation set:
- Loss: 0.5356
- Accuracy: 0.8554
- Precision: 0.8224
- Recall: 0.8722
- F1: 0.8414

### Framework versions

- Transformers 4.42.3
- Pytorch 2.1.2
- Datasets 2.20.0
- Tokenizers 0.19.1
  
## Model description

This model, based on the RoBERTa architecture (roberta-base), is fine-tuned for a sentiment classification task specific to the finance sector. It is designed to classify auditor reports into three sentiment categories: "negative", "neutral", and "positive". This capability can be crucial for financial analysis, investment decision-making, and trend analysis in financial reports.

## Intended uses & limitations

### Intended Uses

This model is intended for professionals and researchers working in the finance industry who require an automated tool to assess the sentiment conveyed in textual data, specifically auditor reports. It can be integrated into financial analysis systems to provide quick insights into the sentiment trends, which can aid in decision-making processes.

### Limitations

- The model is specifically trained on a dataset from the finance domain and may not perform well on general text or texts from other domains.
- The sentiment is classified into only three categories, which might not capture more nuanced sentiments or specific financial jargon fully.
- Like all AI models, this model should be used as an aid, not a substitute for professional financial analysis.

## Training and evaluation data

### Training Data

The model was trained on a proprietary dataset FinanceInc/auditor_sentiment sourced from Hugging Face datasets, which consists of labeled examples of auditor reports. Each report is annotated with one of three sentiment labels: negative, neutral, and positive.

### Evaluation Data

The evaluation was conducted using a split of the same dataset. The data was divided into training and validation sets with a sharding method to ensure a diverse representation of samples in each set.

## Training Procedure

The model was fine-tuned for 5 epochs with a batch size of 8 for both training and evaluation. An initial learning rate of 5e-5 was used with a warm-up step of 500 to prevent overfitting at the early stages of training. The best model was selected based on its performance on the validation set, and only the top two performing models were saved to conserve disk space.

## Evaluation Metrics

Evaluation metrics included accuracy, macro precision, macro recall, and macro F1-score, calculated after each epoch. These metrics helped monitor the model's performance and ensure it generalized well beyond the training data.

## Model Performance

The final model's performance on the test set will be reported in terms of accuracy, precision, recall, and F1-score to provide a comprehensive overview of its predictive capabilities.

## Model Status

This model is currently being evaluated in development.


### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 500
- num_epochs: 5

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy | Precision | Recall | F1     |
|:-------------:|:-----:|:----:|:---------------:|:--------:|:---------:|:------:|:------:|
| 0.4217        | 1.0   | 485  | 0.6358          | 0.8223   | 0.8142    | 0.8135 | 0.8134 |
| 0.6538        | 2.0   | 970  | 0.6491          | 0.8388   | 0.8584    | 0.8025 | 0.8192 |
| 0.3961        | 3.0   | 1455 | 0.5356          | 0.8554   | 0.8224    | 0.8722 | 0.8414 |
| 0.1121        | 4.0   | 1940 | 0.7393          | 0.8512   | 0.8414    | 0.8477 | 0.8428 |
| 0.0192        | 5.0   | 2425 | 0.7233          | 0.8698   | 0.8581    | 0.8743 | 0.8657 |


