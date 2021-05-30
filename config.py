import os
topfolder = '/content/drive/MyDrive/work/coleridge/'

datafolder = f'{topfolder}data/'
modelfolder = f'{topfolder}models/'
outputfolder = f'{topfolder}output/'

def create_subfolders():
    for folder in [datafolder, modelfolder, outputfolder]:
        if not os.path.exists(folder): os.makedirs(folder)

create_subfolders()
