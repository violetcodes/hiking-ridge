
import re
import json

data_f = '/home/vishal_pathak_quantiphi_com/notebooks/coleridge_data/'


def clean_text(txt):
    return re.sub('[^A-Za-z0-9]+', ' ', str(txt).lower())

def get_json(uid, split='train'):
    filepath = data_f + split + '/' + uid + '.json'
    with open(filepath, 'r') as f:
        return json.load(f)
    
json_text = lambda jload: '\n\n\t'.join([i['section_title'] + ': ' + i['text'] for i in jload])

