import os
topfolder = '/content/drive/MyDrive/work/coleridge/'

datafolder = f'{topfolder}data/'
preprocessedfolder = f'{topfolder}preprocessed_data/'
modelfolder = f'{topfolder}models/'
outputfolder = f'{topfolder}output/'


def create_subfolders():
    for folder in [datafolder, modelfolder, outputfolder, preprocessedfolder]:
        if not os.path.exists(folder): os.makedirs(folder)
create_subfolders()

train_csvpath = f'{datafolder}train.csv'
test_csvpath = f'{datafolder}test.csv'
train_folder = f'{datafolder}train/'
test_folder = f'{datafolder}test/'



# prepare data
label_columns_in_csv = ['dataset_title', 'dataset_label', 'cleaned_label']
ner_tagmap = {0:'O', 1:'B', 2:'I'}


finelevel = 'paragraph'
width = 3 # in case finelevel is sentences this is sentence window
overlap = 1
