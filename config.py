import os
topfolder = '/content/drive/MyDrive/work/coleridge'

datafolder = f'{topfolder}/data'
preprocessedfolder = f'{topfolder}/preprocessed_data'
modelfolder = f'{topfolder}/models'
outputfolder = f'{topfolder}/output'


def create_subfolders():
    for folder in [datafolder, modelfolder, outputfolder, preprocessedfolder]:
        if not os.path.exists(folder): os.makedirs(folder)
create_subfolders()

train_csvpath = f'{datafolder}/train.csv'
test_csvpath = f'{datafolder}/test.csv'
train_folder = f'{datafolder}/train'
test_folder = f'{datafolder}/test'



# prepare data
label_columns_in_csv = ['dataset_title', 'dataset_label', 'cleaned_label']



finelevel = 'paragraph'
width = 3 # in case finelevel is sentences this is sentence window
overlap = 1


# training
ner_tags = {0:'O', 1:'B', 2:'I'}
seq_max_len = 256
feature_names = ['fileid', 'labels', 'text']

task = 'ner'
model_name = 'distilbert-base-uncased'
batch_size = 32
use_gpu = True

training_args = dict(
    evaluation_strategy = 'epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=0.01)


# pipeline
split_size = 0.2
