from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from tqdm import tqdm

from data import get_names

names = get_names()

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

y_pred = []

for name in tqdm(names):

    nlp = pipeline("ner", model=model, tokenizer=tokenizer)
    example = f"My name is {name} and I live in Edinburgh"

    ner_results = nlp(example)

    # if no response
    if not ner_results:
        print('hey')
        y_pred.append(0)

    # log 1 if a name was found.
    y_pred_i = 0
    for item in ner_results:
        # https://huggingface.co/dslim/bert-base-NER
        if item['entity'] in ['B-PER', 'I-PER'] and item['index'] == 4:
            y_pred_i = 1

    y_pred.append(y_pred_i)

assert len(y_pred) == len(names)

y_true = np.ones(len(names))
acc = accuracy_score(y_true, y_pred)
print(f'accuracy {acc:.2f}')

df = pd.DataFrame(data={
    'names': names,
    'y_pred': y_pred
})

df.to_csv('data/ner_deep_learning_results.csv')

# accuracy 0.61
