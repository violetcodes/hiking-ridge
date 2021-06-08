import config
import utils
import nltk
# nltk.download('punkt')
from nltk import sent_tokenize


from tqdm.notebook import tqdm as tqdm_notebook
from tqdm import tqdm as tqdm_console

def tqdm(iterable, desc='', total=None):
    try:
        return tqdm_notebook(iterable, desc=desc, total=total)
    except:
        return tqdm_console(iterable, desc=desc, total=total)

def get_file_names_and_labels(df, label_columns=None, fileid_col='Id'):
    '''df containing labels and file name, extract them and json form'''    
    label_columns = label_columns or config.label_columns_in_csv
    fileids = df[fileid_col].unique().tolist()
    labels_proc = lambda x: list({utils.clean_text(i) for i in x})
    
    file_name_and_labels = [
        dict(
            fileid=fileid,
            labels=labels_proc(
                df.loc[df[fileid_col]==fileid, label_columns].values.flatten().tolist())
        ) for fileid in tqdm(fileids)]    
    return file_name_and_labels

def readfile(fileid, split='train'):
    folder = config.train_folder if split=='train' else config.test_folder
    filepath = f'{folder}{fileid}.json' 
    return utils.json_load(filepath)

def text_file(json_file):
    return [(i['section_title'] + ': ' + i['text']) for i in json_file]

def file_loaded_and_label(fileid_list, split='train'):
    '''lazy implementation'''
    for file_dict in fileid_list:        
        filejson = readfile(file_dict['fileid'], split=split)
        texts = text_file(filejson)
        file_dict.update(dict(file_text_list=texts))
        yield file_dict

def finelabel(filedict):
    '''must have file_text_list and labels'''
    file_text_list = filedict['file_text_list']
    if config.finelevel == 'sentence':        
        text = ' '.join(file_text_list)
        texts = sent_tokenize(text)
        o, w, l = config.overlap, config.width, len(texts)
        textlist = [
            ' '.join(texts[max(0, i-o):min(i+w, l)])
            for i in range(0, l, w)]
    else:
        textlist = file_text_list
        
    labelspool = filedict['labels'] # or all labels
    tagged = []        
    for text in textlist:
        text = utils.clean_text(text)
        indexes = utils.findindex(text, labelspool)
        labels_detected = [dict(start=i[0], end=i[1], label=i[2]) for i in indexes]      
        tagged.append(dict(text=text, labels=labels_detected, fileid=filedict.get('fileid', '')))
    return tagged

class CDataset:
    def __init__(self, df):
        self.df = df
        self.fileids = get_file_names_and_labels(df)
        self.filesload()
        # self.tagging()
    
    def filesload(self):
        self.total = len(self.fileids)
        self.file_loaded = file_loaded_and_label(self.fileids)
        # self.file_loaded = [i for i in tqdm(file_loaded_and_label(self.fileids),desc='loading...', total=self.total)]
    def tagged(self):
        for i in self.file_loaded:
            for j in finelabel(i):
                yield j
    
    def tagged_filtered(self):
        for i in self.tagged():
            if i['labels']: yield i 
    
    def tagged_nolabels(self):
        for i in self.tagged():
            if not i['labels']: yield i

    # def tagging(self):
    #     self.tagged = (j for fdict in self.file_loaded for j in finelabel(fdict))
    #     self.tagged_filtered = (i for i in self.tagged if i['labels'])
    #     self.tagged_nolabels = (i for i in self.tagged if not i['labels'])

def pickle_data(cdata, filename):
    utils.pkl_save(cdata, f'{config.preprocessedfolder}{filename}')
    print(f'saved in {config.preprocessedfolder}{filename}')

def load_data(filename):
    return utils.pkl_load(f'{config.preprocessedfolder}{filename}')
    
def highlight_tagged_data(tagged_instance):
    char_list = list(tagged_instance['text'])
    colors = ['white'] * len(char_list)
    rotate_colors = ['blue', 'teal', 'yello', 'pink', 'red']
    j = 0
    for i in tagged_instance['labels']:
        s, e = i['start'], i['end']
        colors[s:e] = [rotate_colors[j]]*(e-s)
        j += 1; j %= len(rotate_colors)
    return utils.highlight_tokens(char_list, colors, joinby='')