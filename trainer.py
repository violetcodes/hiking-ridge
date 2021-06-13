import config 
import utils
import process_data as proc 
from datasets import Dataset, load_metric
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments, Trainer,
    DataCollatorForTokenClassification)

def get_tokenizer(name):
    return AutoTokenizer.from_pretrained(name)
def get_model(name, num_labels, device):
    return AutoModelForTokenClassification.from_pretrained(name, num_labels=num_labels).to(device)



def label_tokens(examples, tokenizer=None):
    '''tokenization for batch of data and NER labels for each token
    '''
    texts = examples['text']
    start_end_offset = [
        [[j['start'], j['end']] for j in i] for i in examples['labels_info']]
    max_len = config.seq_max_len
    tokenizer = tokenizer or get_tokenizer(config.model_name)

    tokenized_data = tokenizer(
        texts, max_length=max_len, truncation=True,
        padding=True, return_offsets_mapping=True)

    labels = np.zeros((len(texts), max_len))

    offsets = np.array(tokenized_data['offset_mapping'])
    for i, ofs in enumerate(offsets):
        cond_b = [False] * max_len
        cond_i = [False] * max_len
        for s, e in start_end_offset[i]:
            cond_b = cond_b | (ofs[..., 0]==s)
            cond_i = cond_i | ((ofs[..., 0]>s) & (ofs[..., 1]<=e))
        labels[i, cond_i] = config.ner_tags['I']
        labels[i, cond_b] = config.ner_tags['B']

    tokenized_data['labels'] = labels.tolist()
    return tokenized_data

def get_mappings(fdata_tagged, feature_names=None):
    '''dictionary flattening
    '''
    feature_names = feature_names or config.feature_names
    mappings = {}
    for i in feature_names:
        mappings[i] = [j[i] for j in fdata_tagged]
    mappings['labels_info'] = mappings['labels']
    del mappings['labels'] # going to store token labels in labels columns 
    # TODO: Fix this later by mapping stored labels to labels_info and updating stored
    return mappings

def get_dataset(fdata_tagged, tokenizer=None):
    mappings = get_mappings(fdata_tagged)
    dataset = Dataset.from_dict(mapping=mappings)
    tokenizer = tokenizer or get_tokenizer(config.model_name)
    dataset1 = dataset.map(lambda x: label_tokens(x, tokenizer), batched=True)
    return dataset1, tokenizer

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    r_ner_tags = {j: i for i, j in config.ner_tags.items()}
    true_predictions = [
        [r_ner_tags[p] for (p, l) in zip(prediction, label) if l!=-100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [r_ner_tags[l] for (p, l) in zip(prediction, label) if l!=-100]
        for prediction, label in zip(predictions, labels)
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return dict(
        precision=results['overall_precision'], 
        recall=results['overall_recall'],
        f1=results['overall_f1'],
        accuracy=results['overall_accuracy']
    )

def get_trainer(
    train_data, eval_data, model=None, tokenizer=None, tr_args=None):
    '''model = transformer model instance
    tokenizer = transformer tokenizer instance
    tr_args = Training arguments
    '''
    device = 'cuda' if config.use_gpu else 'cpu'
    tokenizer = tokenizer or get_tokenizer(config.model_name)
    model = model or get_model(
        config.model_name, num_labels=len(config.ner_tags), device=device)
    
    
    training_args = config.training_args
    training_args.update(tr_args or {})

    args = TrainingArguments(
        'train-ner', **training_args
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


    


