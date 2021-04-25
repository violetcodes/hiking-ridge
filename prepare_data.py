import utils
from tqdm import tqdm

jfile_texts = lambda jload: [i['section_title'] + ': ' + i['text'] for i in jload]
def find_index(text, target, width=100, find_only=False, lowr=True):
    if lowr:
        text = utils.clean_text(text)
        target = utils.clean_text(target)
    
    if find_only:
        return target in text
    
    if target in text:
        start = text.index(target)
        end = start + len(target)
        return start, end
    
    else:
        print('target not found in the text, searching keywords')
        
def keep_longest_labels(labels):
    keep = set()
    lab = {i for i in labels}
    while lab:
        e = lab.pop()
        f = {i for i in lab if e in i or i in e}
        f.add(e)
        l = max(f, key=lambda x: len(x))
        keep.add(l)
        lab.difference_update(f)
    return list(keep)

def get_indx(doc, labels):
    labels = keep_longest_labels(labels)
    return sorted(
        [find_index(doc, label) for label in labels 
            if find_index(doc, label, find_only=True)],
        key=lambda x: x[0])



def split_and_tag(doc, indx):
    # indx = [tuple(i) for i in indx]
    if indx == []:
        return doc.strip().split(), ['O']*len(doc.split())
    tokens = []
    tags = []
    split_points = [0, ] + list({j for i in indx for j in i}) + [len(doc),]
    split_points.sort()    
    
    for s, e in zip(split_points, split_points[1:]):
        
        sp = doc[s:e].strip().split()
        
        # print(s, e, sp)
        if (s, e) in indx:
            tags.extend(['B',] + ['I'] * (len(sp) - 1))
        else : tags.extend(['O']*len(sp))
        tokens.extend(sp)    
    
    return tokens, tags


class DataPrep:
    def __init__(self, id_label_map=None):
        id_lable_map = id_label_map or {}
        self.labels_map = id_label_map
        self.docs_json = {_id: utils.get_json(_id) 
                          for _id in tqdm(self.labels_map, 'reading files')}
        self.docs_text = {_id: jfile_texts(jdoc) 
                          for _id, jdoc in self.docs_json.items()}
        self.meta = {f'{_id}_{i}': dict(text=para, doc_id=_id,
                                              para_no=i, id=f'{_id}_{i}',
                                              clean_text=utils.clean_text(para))
                           for _id, paras in self.docs_text.items()
                           for i, para in enumerate(paras)}
                
        
        
    def prep(self, to_process=None, labels_map=None):
        '''to_process -> list[dict,]: each dict must have clean_text and 
        doc_id
        labels_map: -> dict: doc_id to labels mapping
        '''        
        result = []
        to_process = to_process or list(self.meta.values())
        labels_map = labels_map or self.labels_map
        for doc in tqdm(to_process, 'processing...'):
            text = doc['clean_text']
            doc_id = doc['doc_id']
            labels = labels_map[doc_id]
            indx = get_indx(text, labels)
            if indx != []:
                tokens, tags = split_and_tag(text, indx)
                result.append(dict(
                    tokens=tokens,
                    tags=tags,
                    id=doc['id']
                ))
        return result
