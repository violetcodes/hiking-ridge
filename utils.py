import re
import json
import pickle as pkl

def json_save(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f)
    
def pkl_load(path):
    with open(path, 'rb') as f:
        return pkl.load(f)
    
def pkl_save(obj, path):
    with open(path, 'wb') as f:
        pkl.dump(obj, f)

def clean_text(txt):
    return re.sub('[^A-Za-z0-9]+', ' ', str(txt).lower())

def findindex(text, subtexts):
    return [
        (m.start(), m.end(), subtext) for subtext in subtexts
        for m in re.finditer(subtext, text)]

def json_load(path):
    with open(path, 'r') as f:
        return json.load(f)
