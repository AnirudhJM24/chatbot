import speech_recognition as sr
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import json
import pickle
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import random
from sys import exit
class chatbot:

    with open('intents.json') as f:
        database = json.load(f)
    
    model = keras.models.load_model('chatbot.h5')

    with open('lblencoder.pkl','rb') as f:
        lblencoder = pickle.load(f)
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)



    def __init__(self):
        print("----- your speech to text chatbot starting up ------")
    
    def stt(self):
        speech = sr.Recognizer()
        with sr.Microphone() as m:
            speech.adjust_for_ambient_noise(m)
            print("I am listening")
            aud = speech.listen(m)

        try:
            self.aud_text = speech.recognize_google(aud)
            print("user ------>", self.aud_text)
            if self.aud_text == 'quit':
                exit()
            self.response()
        except :
            print('bot----> Sorry could not understand')
    def response(self):
        key = self.inference()
        for i in self.database['intents']:
            if i['tag'] == key:
                print('bot ----->',np.random.choice(i['responses']) )


    def inference(self):
        sequence = self.tokenizer.texts_to_sequences([self.aud_text])
        test = pad_sequences(sequence, maxlen=20)
        #print(self.lblencoder.inverse_transform([np.argmax(self.model.predict(test))]))
        return self.lblencoder.inverse_transform([np.argmax(self.model.predict(test))])
    


if __name__ == '__main__':

    cb = chatbot()

    while True:
        cb.stt()
