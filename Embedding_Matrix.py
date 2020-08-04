from numpy import asarray, zeros
from prepare_for_embedding import prepare_for_embedding_layer
# Create a dictionary containing words which corresponding feature values
def create_embedding_dict():
    embeddings_dict = dict()
    glove_file = open('glove.6B.100d.txt', encoding="utf8")

    for line in glove_file:
        records = line.split()
        word = records[0]
        coef = asarray(records[1:], dtype='float32')
        embeddings_dict[word] = coef
    glove_file.close()
    return embeddings_dict

# Create embedding matrix with shape of (vocab_size, max_len)
# Embedding matrix is the product of (vocab_size, vocab_size) and (vocab_size, max_len)
def create_embedding_matrix(embeddings_dictionary, tokenizer, vocab_size, max_len):
    embedding_matrix = zeros((vocab_size, max_len))
    for word, index in tokenizer.word_index.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector

    return embedding_matrix