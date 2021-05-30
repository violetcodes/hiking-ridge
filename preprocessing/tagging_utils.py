def findindex(text, subtexts):
    return [
        (m.start(), subtexts) for subtext in subtexts
        for m in re.finditer(subtext, text)]

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

def split_and_tag(text, indx):
    if indx == []:
        return text.strip().split(), [0]*len(text.split())
    tokens = []
    tags = []
    split_points = [0, ] + list({j for i in indx for j in i}) + [len(text),]
    split_points.sort()    
    
    for s, e in zip(split_points, split_points[1:]):
        sp = text[s:e].strip().split()
        if (s, e) in indx:
            tags.extend([1,] + [2] * (len(sp) - 1))
        else : tags.extend([0]*len(sp))
        tokens.extend(sp)    
    
    return tokens, tags
