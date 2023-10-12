import random
import json
import pickle
from tkinter import Y
import numpy as np
import nltk 
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation , Dropout
from keras.optimizers import gradient_descent_v2



gradient_descent_v2.SGD()
lemmetizer = WordNetLemmatizer()
intents = json.loads(open("intents.json").read())


words = []
classes = []
documents = []
ignore_letters = ['?','!','.',",",]

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent ['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmetizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

classes = sorted(set(classes))
 

pickle.dump(words, open('words.pkl','wb'))
pickle.dump(words, open('classes.pkl','wb'))

training = []
output_empty = [0]* len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmetizer.lemmatize(word.lower()) for word in word_patterns]
    for words in words:
        bag.append(1) if words in word_patterns else bag.append(0)

        output_row = list(output_empty)
        output_row[classes.index(document[1])] = 1
        training.append([bag,output_row])

random.shuffle
training = np.array(training)

train_x = list(training [:, 0])
train_y = list(training [:, ])

output_row = list(output_empty)
output_row[classes.index(document[1])] = 1
training.append(bag)
train_y.append(output_row)

training = np.array(training)
train_y = np.array(train_y)
print(training.shape, train_y.shape)

model = Sequential()
model.add(Dense(128,input_shape=(len(train_x[0]),),activation= 'relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]),activation='softmax'))


sgd = gradient_descent_v2.SGD(learning_rate=0.01,decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer = sgd, metrics=['accuracy'])


#model.fit(np.asarray([train_x]), batch_size = 5, epochs =200, verbose = 1)
# model.fit(np.asarray([train_y]), batch_size = 5, epochs =200, verbose = 1)
model.fit(np.array(train_x), np.array(train_y), batch_size = 5, epochs =200, verbose = 1)


model.save('chatbot_model.model')
print('done')