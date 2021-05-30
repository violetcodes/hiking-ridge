import pandas as pd 
import config as cfg
import utils
from collections import defaultdict
from preprocessing.tagging_utils import *
import time, os, re

def readfile(fileid, split='train'):
    folder = cfg.train_folder if split=='train' else cfg.test_folder
    filepath = f'{folder}{fileid}.json' 
    return utils.read_json(filepath)

def cleaned_text_fromjson(json_file):
    return [utils.clean_text(i['section_title'] + ': ' + i['text']) for i in json_file]


class Preprocess:
    '''data annotation class'''
    def __init__(self, df, split='train'):
        self.df = df
        self.fileids = self.df.Id.unique().tolist()
        self.split = split 
        self.files = {}
        self.cleanfiles = {}
        self.tagmap = cfg.ner_tagmap
        self.processed_data = []
        self.getlabelmap()
    
    def getlabelmap(self):
        df = self.df
        self.cols = cols = cfg.labelcols        
        label_map = defaultdict(set)
        for fileid in self.fileids:
            label_map = {
                j1 for j in df[df.Id==fileid][cols].values
                for j1 in j
            }
        self.label_map = label_map        
    
    def process(self):
        '''gives iterator for processed files'''
        for fileid in self.fileids:
            self.files[fileid] = jsonfile = readfile(fileid)
            self.cleanfiles[fileid] = clean_text = cleaned_text_fromjson(jsonfile)

            labels = keep_longest_labels(
                [utils.clean_text(i) for i in self.label_map[fileid]])
            for i, cltx in enumerate(clean_text):
                idx_with_labels = findindex(cltx, labels)
                idx = [i for i, j in idx_with_labels]
                found_labels = {j for i, j in idx_with_labels}

                tokens, tags = split_and_tag(cltx, idx)
                s = dict(
                    fileid=fileid,
                    passageid=i,
                    text=cltx,
                    labels=found_labels,
                    tokens=tokens,
                    tags=tags,
                    indx=inx,
                    has_tags=(1 in tags or 2 in tags)
                )
                self.processed_data.append(s)
                yield s

    def save_as_json(self, savedir=None, name=''):
        savedir = savedir or cfg.preprocessedfolder
        init_dict = dict(
            df=self.df.to_dict(),
            fileids=self.fileids,
            label_map=self.label_map,
            cleanfiles=self.cleanfiles,
            tagmap=self.tagmap,
            time=time.ctime(),
            name=name,
            split=self.split
        )

        utils.json_save(init_dict, f'{savedir}info_{name}.json')
        taggedfiles_location = f'{savedir}tagged_jsonfiles_{name}' 
        if not os.path.exists(taggedfiles_location):
            os.makedirs(taggedfiles_location)
        for i in self.processed_data:
            jname = f"{i['fileid']}_{i['passageid']}.json"
            utils.json_save(i, f'{taggedfiles_location}/{jname}')
        
        print('Saved processed data')
        
    def load_from_json(self, path=None, name=''):
        path = path or cfg.preprocessedfolder
        init_json = utils.read_json(f'{path}info_{name}.json')
        df = init_json.pop('df')
        self.df = pd.from_dict(df)
        for i in init_json:
            setattr(self, i, init_json[i])
        
        processed_data = []
        taggedfiles_location = f'{path}tagged_jsonfiles_{name}'
        for i in os.listdir(taggedfiles_location):
            processed_data.append(utils.read_json(f'{taggedfiles_location}/{i}'))
        self.processed_data = processed_data

        print('Loaded processed data')
        





                






        
    





