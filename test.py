from preprocessing import train_ready as t

def test_get_file_names_and_labels():
    import pandas as pd 
    import random
    s = [random.choices(range(100,104), k=4) for i in range(5)]
    df = pd.DataFrame(s, columns=['id', 'lab1', 'lab2', 'lab3']).astype(str)
    print(df)
    label_columns = ['lab1', 'lab2',]
    fileid_col = 'id'
    
    print(t.get_file_names_and_labels(df, label_columns=label_columns, fileid_col=fileid_col))

test_get_file_names_and_labels()