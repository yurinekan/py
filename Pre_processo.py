import pandas as pd
from nltk.stem import WordNetLemmatizer, RSLPStemmer
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
import re
import random
import nltk
import pickle as pkl
from nltk.tokenize import word_tokenize, RegexpTokenizer
import numpy as np

def vocabulario(context, utterance):
    tokenizer = Tokenizer(filters='"#$%&()*+,-./:;<=>[\]^_`{|}~ ?!', lower=True, split=' ',
                          char_level=False, oov_token='NotFound')
    tokenizer.fit_on_texts(context+utterance)
    tokens = tokenizer.word_index
    print("Vocab:", tokens)
    # output = open('Salvo\\tokenizer.pkl', 'wb')
    # pkl.dump(tokenizer, output)
    # output.close()
    return tokenizer

def gerando_sequencia_numerica(tokenizer, texto, tamanho_sentenca):
    texto = tokenizer.texts_to_sequences(texto)
    texto = pad_sequences(texto, tamanho_sentenca, padding='post')
    print('Exemplo padded: ', texto[0])
    return texto

def tokenizacao_por_palavra(corpus):
    new_corpus = []
    for l in corpus:
        new_corpus.append(word_tokenize(l))
    print("Exemplos tokenizados: ", new_corpus[0])
    return new_corpus

def remove_stopwords(text):
    stopwords = nltk.corpus.stopwords.words('portuguese')
    stopwords[:10]
    new_text = []
    for sentence in text:
        #print(sentence)
        sentence = [word.lower() for word in sentence if word not in stopwords]
        #print(sentence)
        new_text.append(sentence)
    return new_text

def tirando_media_sentencas(corpus):
    soma = 0
    for sentenca in corpus:
        soma = soma + len(sentenca.split(' '))
    media = int(soma/len(corpus))
    print('Media das sentenças: ', media)
    # output = open('Salvo\\media.pkl', 'wb')
    # pkl.dump(media, output)
    # output.close()
    return media

def voltando_string(corpus):
    new_corpus = []
    for sentence in corpus:
        new_sentence = " ".join(sentence)
        new_sentence = new_sentence.lstrip()
        new_corpus.append(new_sentence)
    print('Text: ', new_corpus[0])
    return new_corpus

