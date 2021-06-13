
def get_annotated_dataset(name, n=None):
    '''load data from pickle
    Args:
        name: str = name of the file e.g. example.pkl
    Returns:
    '''
    import config, utils
    tagged = utils.json_load(config.preprocessedfolder + '/' + name)
    # import process_data as proc

    # Pdata = proc.load_data(name)
    tagged = tagged[:n]
    return tagged 

def train(dataname, return_trainer=False):
    processed_data = get_annotated_dataset(dataname)
    from sklearn.model_selection import train_test_split
    import trainer 

    train_data, eval_data = train_test_split(processed_data, test_size=config.split_size)

    train_data, eval_data = list(map(trainer.get_dataset, [train_data, eval_data]))
    tr = trainer.get_trainer(train_data, eval_data)
    
    if return_trainer:
        return tr 
    
    tr.train()

    preds_prob, labels, _ = tr.predict(eval_data)

    predictions = np.argmax(preds_prob, axis=2)

    metrics = trainer.compute_metrics((preds_prob, labels))
    return tr, metrics, predictions

  


