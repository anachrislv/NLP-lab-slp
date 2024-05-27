from torch.utils.data import Dataset
from tqdm import tqdm

import numpy as np
from config import MAX_SENTENCE_SIZE
from nltk.tokenize import word_tokenize

class SentenceDataset(Dataset):
    """
    Our custom PyTorch Dataset, for preparing strings of text (sentences)
    What we have to do is to implement the 2 abstract methods:

        - __len__(self): in order to let the DataLoader know the size
            of our dataset and to perform batching, shuffling and so on...

        - __getitem__(self, index): we have to return the properly
            processed data-item from our dataset with a given index
    """

    def __init__(self, X, y, word2idx):
        """
        In the initialization of the dataset we will have to assign the
        input values to the corresponding class attributes
        and preprocess the text samples

        -Store all meaningful arguments to the constructor here for debugging
         and for usage in other methods
        -Do most of the heavy-lifting like preprocessing the dataset here


        Args:
            X (list): List of training samples
            y (list): List of training labels
            word2idx (dict): a dictionary which maps words to indexes
        """

        # self.data = X
        # self.labels = y
        # self.word2idx = word2idx

        # EX2
        X = [[w for w in word_tokenize(x)] for x in X]
        self.data = X
        self.labels = y
        self.word2idx = word2idx    
        # raise NotImplementedError

    def __len__(self):
        """
        Must return the length of the dataset, so the dataloader can know
        how to split it into batches

        Returns:
            (int): the length of the dataset
        """

        return len(self.data)

    def __getitem__(self, index):
        """
        Returns the _transformed_ item from the dataset

        Args:
            index (int):

        Returns:
            (tuple):
                * example (ndarray): vector representation of a training example
                * label (int): the class label
                * length (int): the length (tokens) of the sentence

        Examples:
            For an `index` where:
            ::
                self.data[index] = ['this', 'is', 'really', 'simple']
                self.target[index] = "neutral"

            the function will have to return something like:
            ::
                example = [  533  3908  1387   649   0     0     0     0]
                label = 1
                length = 4
        """

        # EX3

        # return example, label, length
        max_length = MAX_SENTENCE_SIZE

        def adjust(vector):
            global length
            if len(vector) == max_length:
                length = max_length
                return vector

            if len(vector) > max_length:
                # Crop
                length = max_length
                return vector[:max_length]

            if len(vector) < max_length:
                # Pad
                length = len(vector)
                zeroes = [0 for _ in range(max_length - len(vector))]
                return vector + zeroes

        sentence = self.data[index]
        word2idx = self.word2idx

        example = [word2idx[word] if word in word2idx else word2idx["<unk>"] for word in sentence]
        example = adjust(example)

        example = np.array(example)

        label = self.labels[index]

        return example, label, length
        # raise NotImplementedError

