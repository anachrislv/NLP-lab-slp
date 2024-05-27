import os
import warnings
import io
import sys

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from training import train_dataset, eval_dataset, torch_train_val_split, get_metrics_report

import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from config import EMB_PATH, MAX_SENTENCE_SIZE
from dataloading import SentenceDataset
from models import BaselineDNN, LSTM
from attention import SimpleSelfAttentionModel, MultiHeadAttentionModel, TransformerEncoderModel
from training import train_dataset, eval_dataset
from utils.load_datasets import load_MR, load_Semeval2017A
from utils.load_embeddings import load_word_vectors
from early_stopper import EarlyStopper

from utils.load_datasets import load_MR, load_Semeval2017A
from utils.load_embeddings import load_word_vectors

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

########################################################
# Configuration
########################################################


# Download the embeddings of your choice
# for example http://nlp.stanford.edu/data/glove.6B.zip

# 1 - point to the pretrained embeddings file (must be in /embeddings folder)
EMBEDDINGS = os.path.join(EMB_PATH, "glove.6B.50d.txt")

# 2 - set the correct dimensionality of the embeddings
EMB_DIM = 50

EMB_TRAINABLE = False
BATCH_SIZE = 128
EPOCHS = 50

# def load_vectors(fname):
#     fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
#     n, d = map(int, fin.readline().split())
#     data = {}
#     for line in fin:
#         tokens = line.rstrip().split(' ')
#         data[tokens[0]] = map(float, tokens[1:])
#     return data


#DATASET = "MR"   options: "MR", "Semeval2017A"

# if your computer has a CUDA compatible gpu use it, otherwise use the cpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET = "MR"

########################################################
# Define PyTorch datasets and dataloaders
########################################################

# load word embeddings
print("loading word embeddings...")
word2idx, idx2word, embeddings = load_word_vectors(EMBEDDINGS, EMB_DIM)

# load the raw data
if DATASET == "Semeval2017A":
    X_train, y_train, X_test, y_test = load_Semeval2017A()
elif DATASET == "MR":
    X_train, y_train, X_test, y_test = load_MR()
else:
    raise ValueError("Invalid dataset")

# convert data labels from strings to integers
le = LabelEncoder()
y_train = le.fit_transform(y_train)   # EX1
y_test = le.fit_transform(y_test)   # EX1
n_classes = le.classes_.size  # EX1 - LabelEncoder.classes_.size

# Define our PyTorch-based Dataset
train_set = SentenceDataset(X_train, y_train, word2idx)
test_set = SentenceDataset(X_test, y_test, word2idx)

train_loader, val_loader = torch_train_val_split(
    train_set, batch_train=BATCH_SIZE, batch_eval=BATCH_SIZE, val_size=0.2, shuffle=True, seed=420
)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

#############################################################################
# Model Definition (Model, Loss Function, Optimizer)
#############################################################################

models_dict = {
    "LSTM":
        LSTM(output_size=n_classes,  
                    embeddings=embeddings,
                    trainable_emb=EMB_TRAINABLE),

    "BLSTM":
        LSTM(output_size=n_classes,  
                    embeddings=embeddings,
                    trainable_emb=EMB_TRAINABLE,
                    bidirectional=True),

    "SSA":
        SimpleSelfAttentionModel(output_size=n_classes,  # EX8
            embeddings=embeddings),
    "MHA":
        MultiHeadAttentionModel(output_size=n_classes,  # EX8
            embeddings=embeddings),
    "ΤrΕn":
        TransformerEncoderModel(output_size=n_classes,  # EX8
            embeddings=embeddings),
}

for modelname in models_dict:
    model = models_dict[modelname]

    # move the mode weight to cpu or gpu
    model.to(DEVICE)

    # We optimize ONLY those parameters that are trainable (p.requires_grad==True)
    criterion = torch.nn.BCEWithLogitsLoss() if n_classes == 2 else torch.nn.CrossEntropyLoss()   # EX8
    parameters = filter(lambda l: l.requires_grad, model.parameters())  # EX8
    optimizer = torch.optim.Adam(parameters, lr=0.001)  # EX8

    early_stopper = EarlyStopper(model, save_path=f'models/best_{modelname}.pth', patience=5, min_delta=0.0)

    train_losses = []
    test_losses = []
        
    for epoch in range(1, EPOCHS + 1):
        # train the model for one epoch
        train_loss = train_dataset(epoch, train_loader, model, criterion, optimizer, DATASET)

        # evaluate the performance of the model
        val_loss, (y_train_gold, y_train_pred) = eval_dataset(val_loader,
                                                                model,
                                                                criterion,
                                                                DATASET)
        # Append the loss (for plotting)
        train_losses.append(train_loss)

        #if 1 or modelname == "LSTM" or modelname == "BLSTM":
        if early_stopper.early_stop(val_loss):
            print("Early stopping triggered. Training stopped.")
            break

    # Load the best model
    model.load_state_dict(torch.load(f'models/best_{modelname}.pth'))

    # Evaluation on the test set
    test_loss, (y_test_gold, y_test_pred) = eval_dataset(test_loader, model, criterion, DATASET)

    for tr, list in zip(["train"], [train_losses, test_losses]):
        plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.plot(list)
        plt.savefig(f"outputs/{modelname}_{DATASET}_{tr}_loss.pdf")

    #############################################################################
    # Evaluation
    #############################################################################

    for (golds, preds), traintest in zip([(y_train_gold, y_train_pred), (y_test_gold, y_test_pred)], ["train", "test"]):
        accuracy = 0
        recall = 0
        f1_score = 0

        for gold, pred in zip(golds, preds):
            gold = gold.cpu().numpy()  # Move tensor to CPU and convert to NumPy array
            pred = pred.cpu().numpy() 
                
            accuracy += metrics.accuracy_score(gold, pred)
            recall +=  metrics.recall_score(gold, pred, average="macro")
            f1_score += metrics.f1_score(gold, pred, average="macro")

        accuracy /= len(golds)
        recall /= len(golds)
        f1_score /= len(golds)

        print(f"\nEvaluation Metrics for {modelname}_{traintest.upper()} dataset\n")
        print(f"Accuracy: {accuracy}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1_score}")

        with open(f"outputs/{modelname}_{DATASET}_{traintest}.txt", "w") as outfile:
            print(f"Accuracy: {accuracy}", file=outfile)
            print(f"Recall: {recall}", file=outfile)
            print(f"F1 Score: {f1_score}", file=outfile)
