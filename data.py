import numpy as np
import pandas as pd
from datasets import load_dataset


def get_names():
    names = pd.read_csv('data/baby_names.csv')
    names = names.drop_duplicates()
    names = names.to_numpy().flatten()
    names = [name.lower() for name in names]
    names = list(set(list(names)))
    return names

def get_coll2003_tokens():

    dataset = load_dataset("conll2003")['train'].to_pandas()

    tokens = np.concatenate(dataset.tokens, axis=0)
    tokens = [token.lower() for token in tokens]
    tokens = list(set(list(tokens)))

def get_names_coll2003_overlap():

    names = get_names()
    tokens = get_coll2003_tokens()

    common = list(set(names) & set(tokens))

    n = len(common) / len(names)

    print(n)

def get_counts():
    names = pd.read_csv('data/baby_name_counts.csv')
    print(names)

if __name__ == '__main__':

    c = get_counts()


