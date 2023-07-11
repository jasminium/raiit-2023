import pandas as pd
from tqdm import tqdm
import json
from uuid import uuid4
from collections import Counter


from datasets import load_dataset

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

import optuna



def process(split='train'):    
   
    utterance = []
    ids = []
    label = []
    act = []
    
    # Apply the function to all examples in the dataset
    dataset = load_dataset('daily_dialog', split=split)
    
    for i in tqdm(range(len(dataset))):
        example = dataset[i]
        did = uuid4()
        for j in range(len(example['dialog'])):
            text = example['dialog'][j]
            # add previous sentnce xontext
            #if j > 1:
            #    text = str(example['emotion'][j - 1]) + ' ' + example['dialog'][j - 1] + ' ' + text
            utterance.append(example['dialog'][j])
            act.append(example['act'][j])
            label.append(example['emotion'][j])
            ids.append(did)

    data = {
        'text': utterance,
        'label': label,
        'attr': act,
        'id': ids
    }

    df = pd.DataFrame(data=data)

    return df

df_train = process(split='train')
print('n train', len(df_train))
df_valid = process(split='validation')
df_test = process(split='test')

# improves macro f1
#rus = RandomOverSampler(random_state=42)
#df_train, _ = rus.fit_resample(df_train, df_train.label)

counts = Counter(df_train.label)
print('train label dist.', counts)


def objective(trial):
    
    # hyper params
    alpha = trial.suggest_float('alpha', 1e-5, 1e-3, log=True)

    clf = SGDClassifier(loss='log_loss', penalty='l2', alpha=alpha, n_jobs=-1)
    #clf = RandomForestClassifier(n_estimators=200, max_depth=200)

    count_vect = CountVectorizer()

    X_train_counts = count_vect.fit_transform(df_train.text.to_list())
    X_valid_counts = count_vect.transform(df_valid.text.to_list())
    X_test_counts = count_vect.transform(df_test.text.to_list())

    tf_transformer = TfidfTransformer(use_idf=True).fit(X_train_counts)
    X_train_tfidf = tf_transformer.transform(X_train_counts)
    X_valid_tfidf = tf_transformer.transform(X_valid_counts)
    X_test_tfidf = tf_transformer.transform(X_test_counts)

    clf.fit(X_train_tfidf, df_train.label)

    y_pred = clf.predict(X_valid_tfidf)
    y_true = df_valid.label
    report = classification_report(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='micro')
    return f1

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)
study.best_params  # E.g. {'x': 2.002108042}
print('f1', study.best_value, study.best_params) # alpha 1e-5


# test


clf = SGDClassifier(loss='log_loss', penalty='l2', alpha=study.best_params['alpha'], n_jobs=-1)
#clf = RandomForestClassifier(n_estimators=200, max_depth=200)

count_vect = CountVectorizer()

X_train_counts = count_vect.fit_transform(df_train.text.to_list())
X_valid_counts = count_vect.transform(df_valid.text.to_list())
X_test_counts = count_vect.transform(df_test.text.to_list())

tf_transformer = TfidfTransformer(use_idf=True).fit(X_train_counts)
X_train_tfidf = tf_transformer.transform(X_train_counts)
X_valid_tfidf = tf_transformer.transform(X_valid_counts)
X_test_tfidf = tf_transformer.transform(X_test_counts)

clf.fit(X_train_tfidf, df_train.label)

y_pred = clf.predict(X_test_tfidf)
y_true = df_test.label
report = classification_report(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='micro')

print(report)
print(f1)