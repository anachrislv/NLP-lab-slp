from transformers import pipeline
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from utils.load_datasets import load_MR, load_Semeval2017A
from training import get_metrics_report

DATASETS = ['MR', 'Semeval2017A']
PRETRAINED_MODELS_MR = ['siebert/sentiment-roberta-large-english', 'distilbert-base-uncased-finetuned-sst-2-english', 'textattack/bert-base-uncased-imdb']

PRETRAINED_MODELS_SEM = ['cardiffnlp/twitter-roberta-base-sentiment', 'finiteautomata/bertweet-base-sentiment-analysis', 'finiteautomata/beto-sentiment-analysis']


LABELS_MAPPING = {
    'siebert/sentiment-roberta-large-english': {
        'POSITIVE': 'positive',
        'NEGATIVE': 'negative',
    },
    'distilbert-base-uncased-finetuned-sst-2-english': {
        'POSITIVE': 'positive',
        'NEGATIVE': 'negative',
    },
    'textattack/bert-base-uncased-imdb': {
        'LABEL_1': 'positive',
        'LABEL_0': 'negative',
    },
    'cardiffnlp/twitter-roberta-base-sentiment': {
        'LABEL_0': 'negative',
        'LABEL_1': 'neutral',
        'LABEL_2': 'positive',
    },
    'finiteautomata/bertweet-base-sentiment-analysis': {
        'NEG': 'negative',
        'NEU': 'neutral',
        'POS': 'positive',
    },
    'finiteautomata/beto-sentiment-analysis': {
        'NEG': 'negative',
        'NEU': 'neutral',
        'POS': 'positive',
    }
}

if __name__ == '__main__':
    for DATASET in DATASETS:
        # load the raw data
        if DATASET == "Semeval2017A":
            X_train, y_train, X_test, y_test = load_Semeval2017A()
            PRETRAINED_MODELS = PRETRAINED_MODELS_SEM
        elif DATASET == "MR":
            X_train, y_train, X_test, y_test = load_MR()
            PRETRAINED_MODELS = PRETRAINED_MODELS_MR
        else:
            raise ValueError("Invalid dataset")

        # encode labels
        le = LabelEncoder()
        le.fit(list(set(y_train)))
        y_train = le.transform(y_train)
        y_test = le.transform(y_test)
        n_classes = len(list(le.classes_))

        # define a proper pipeline
        for PRETRAINED_MODEL in PRETRAINED_MODELS:
            sentiment_pipeline = pipeline("sentiment-analysis", model=PRETRAINED_MODEL)

            y_pred = []
            for x in tqdm(X_test):
                # TODO: Main-lab-Q6 - get the label using the defined pipeline 
                result = sentiment_pipeline(x)[0]
                label = result['label']       
                y_pred.append(LABELS_MAPPING[PRETRAINED_MODEL][label])

            y_pred = le.transform(y_pred)
            print(f'\nDataset: {DATASET}\nPre-Trained model: {PRETRAINED_MODEL}\nTest set evaluation\n{get_metrics_report([y_test], [y_pred])}')
