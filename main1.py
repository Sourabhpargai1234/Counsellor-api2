import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Sample data - more comprehensive
sentences = [
    'I love this movie', 'I hate this movie', 'This was an amazing experience',
    'I am not a fan of this', 'Absolutely fantastic!', 'Not my cup of tea',
    'This is a great day', 'I feel terrible', 'This is so bad', 'I am very happy',
    'I am extremely sad', 'This is the best!', 'I don’t like this', 'I am delighted',
    'I am disappointed', 'It’s a bad day today', 'I had a great time', 'I am upset',
    'What a wonderful experience', 'I regret coming here'
]

labels = [1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0]  # 1: positive, 0: negative

# Tokenizing and padding
tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
padded_sequences = pad_sequences(sequences, padding='post')

# Model definition
model = Sequential([
    Embedding(100, 16, input_length=padded_sequences.shape[1]),
    GlobalAveragePooling1D(),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(padded_sequences, np.array(labels), epochs=30)

# Save the model
model.save('sentiment_model.h5')

# Save tokenizer
import pickle
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
