import random
import json
import pickle
import numpy as np
import nltk
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences


lemmatizer = WordNetLemmatizer()
intents = json.loads(open("/Users/akhil/Desktop/Programming /ai chat bot /intents.json").read())

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ","]

for intent in intents['intents']:
    for intent in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

tokenizer = Tokenizer(num_words=len(words), filters=ignore_letters)
tokenizer.fit_on_texts([pattern for pattern, tag in documents])
sequences = tokenizer.texts_to_sequences([pattern for pattern, tag in documents])
max_sequence_length = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')

training = []
output_empty = [0] * len(classes)

for index, padded_sequence in enumerate(padded_sequences):
    output_row = list(output_empty)
    output_row[classes.index(documents[index][1])] = 1
    training.append([padded_sequence, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, -1])

model = Sequential()
model.add(Dense(128, input_shape=(max_sequence_length,), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(train_x, train_y, batch_size=5, epochs=200, verbose=1) # type: ignore
model.save('chatbot_model.h5', hist)
print('done')

def preprocess_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token not in ignore_letters]
    padded_sequence = pad_sequences(tokenizer.texts_to_sequences([tokens]), maxlen=max_sequence_length, padding='post')
    return padded_sequence

def predict_intent(text):
    padded_sequence = preprocess_text(text)
    predictions = model.predict(padded_sequence)
    predicted_class_index = np.argmax(predictions)
    predicted_class = classes[predicted_class_index]
    return predicted_class
