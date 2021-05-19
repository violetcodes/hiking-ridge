import utils

import transformers
from transformers import AutoTokenizer
from transformers import (
    AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification)
from datasets import Dataset, load_metric

from sklearn.model_selection import train_test_split

task = 'ner'
model_checkpoint = 'distilbert-base-uncased'
batch_size = 4

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

ner_tags = dict(O=0, B=1, I=2)
r_ner_tags = {j: i for i, j in ner_tags.items()}
metric = load_metric('seqeval')

model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint, num_labels=len(ner_tags))

def tokenize_and_align(examples):
    tokenized_inputs = tokenizer(
        examples['tokens'], padding='max_length',
        max_length=512, truncation=True,
        is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples['ner_tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            else: label_ids.append(ner_tags[label[word_idx]])

        labels.append(label_ids)
    
    tokenized_inputs['labels'] = labels
    return tokenized_inputs

def read_prepared_data(path):
    tagged_data = utils.read_pkl(path)
    mapping = dict(
        id=[i['id'] for i in tagged_data],
        tokens = [i['tokens'] for i in tagged_data],
        ner_tags = [i['tags'] for i in tagged_data]
    )
    data = Dataset.from_dict(mapping=mapping)
    return data.map(tokenize_and_align, batched=True)

def compute_metrics(p):
    predictions, lables = p
    true_predictions = [
        r_ner_tags[p] for (p, l) in zip(prediction, label) if l!=-100
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        r_ner_tags[l] for (p, l) in zip(prediction, label) if l!=-100
        for prediction, label in zip(predictions, labels)
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return dict(
        precision=results['overall_precision'], 
        recall=results['overall_recall'],
        f1=results['overall_f1'],
        accuracy=results['overall_accuracy']
    )


def get_trainer(train_data, eval_data, model=None):
    model = model or AutoModelForTokenClassification.from_pretrained(
        model_checkpoint, num_labels=len(ner_tags))
    
    args = TrainingArguments(
        'train-ner',
        evaluation_strategy = 'epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    data_collector = DataCollatorForTokenClassification(tokenizer)

    return Trainer(
        model,
        args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=data_collector,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )



    
if __name__ == "__main__":
    data_path = 'path'

    data = read_data(data_path)
    train_data, eval_data = train_test_split(data)
    
    trainer = get_trainer(train_data, eval_data)

    print('Going to train')
    trainer.train()

    trainer.save_model(utils.data_f)
    


    
    






