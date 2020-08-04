from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle

# Prepare for Embedding layer
# Use tokenizer to convert word to unique index
def prepare_for_embedding_layer(data, x_train, x_test):
    tokenizer = Tokenizer()
    # The method below creates the vocabulary index based on word frequency. So lower integer means more frequent word
    tokenizer.fit_on_texts(x_train)

    # Convert text to list of integers
    x_train = tokenizer.texts_to_sequences(x_train)
    x_test = tokenizer.texts_to_sequences(x_test)

    # We need to have a fixed size for word embedding in order to feed it into Embedding layer
    vocab_size = len(tokenizer.index_word) + 1
    max_len = 100

    # Use post for padding after the sentence up to the max length
    x_train = pad_sequences(sequences=x_train, maxlen=max_len, padding='post')
    x_test = pad_sequences(sequences=x_test, maxlen = max_len, padding='post')

    # Use a dictionary to store some parameters.
    parameters = dict()
    parameters['tokenizer'] = tokenizer
    parameters['vocab_size'] = vocab_size
    parameters['max_len'] = max_len

    file = open('parameters.pickle', 'wb')
    pickle.dump(parameters, file)
    file.close()

    return parameters, x_train, x_test