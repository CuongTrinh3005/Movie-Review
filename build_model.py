from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Flatten

def build_model(vocab_size, embedding_matrix, max_len, input_shape):
    model = Sequential()
    embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=max_len, trainable=False)
    model.add(embedding_layer)
    # model.add(Flatten())
    model.add(LSTM(128, input_shape=input_shape, dropout=0.2, return_sequences=False))

    # model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    model.summary()
    return model