
import re
import json

data_f = '/home/vishal_pathak_quantiphi_com/notebooks/coleridge_data/'


def clean_text(txt):
    return re.sub('[^A-Za-z0-9]+', ' ', str(txt).lower())

def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def get_json(uid, split='train'):
    filepath = data_f + split + '/' + uid + '.json'
    return read_json(filepath)
    
json_text = lambda jload: '\n\n\t'.join([i['section_title'] + ': ' + i['text'] for i in jload])
