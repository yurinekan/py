from keras.models import Sequential, Model, load_model
from keras.layers import Embedding, LSTM, Dense, Flatten, Input, Multiply, Activation, Add
import Pre_processo as pp
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from gensim.models import Word2Vec
from collections import defaultdict
import pickle as pkl
from nltk.tokenize import sent_tokenize, word_tokenize

def criando_embeddings_matrix(tokenizer, embeddings):
    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, 100))
    not_found = 0
    for word, i in tokenizer.word_index.items():
        try:
            embedding_vector = embeddings[word]
        except:
            embedding_vector = None

        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            not_found+=1
    print('Palavras sem embeddings: ', not_found)
    #print(embedding_matrix[0:20])
    print('Tamanho da matriz de embeddings: ', len(embedding_matrix))
    # output = open('Salvo\\embedding_matrix.pkl', 'wb')
    # pkl.dump(embedding_matrix, output)
    # output.close()
    return embedding_matrix

def encoder_context_utterance(vocab_size, tamanho_sentenca, embedding_matrix):
    encoder = Sequential()
    encoder.add(Embedding(vocab_size+1, output_dim=100, input_length=tamanho_sentenca, weights=[embedding_matrix], trainable=False))#, trainable=False
    encoder.add(LSTM(units=150, activation='tanh', recurrent_dropout=0.2, dropout=0.2, return_sequences=False))
    return encoder

def dual_lstm(encoder, tamanho_sentenca):
    print("-----------MODE-----------")
    print()
    context_input = Input(shape=(tamanho_sentenca,), dtype='int32')
    utterance_input = Input(shape=(tamanho_sentenca,), dtype='int32')

    context_branch = encoder(context_input)
    print("context_branch: ", context_branch)
    utterance_branch = encoder(utterance_input)
    print("utterance_branch: ", utterance_branch)

    dense_context = Dense(150, activation='linear')(context_branch)
    #concatenated = Add()([context_branch, utterance_branch])
    multi = Multiply()([dense_context, utterance_branch])
    print("multi: ", multi)
    tanh = Activation('tanh')(multi)
    print("Tanh: ", tanh)
    out = Dense(1, activation='sigmoid')(tanh)

    dual_encoder = Model([context_input, utterance_input], out)
    dual_encoder.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    print()

    return dual_encoder

def test_model(dual_lstm, context, utterance, labels):
    a_preds = dual_lstm.predict([context, utterance])
    print(a_preds[0:5])

    y_pred = []
    for pred in a_preds:
        if pred > 0.5:
            y_pred.append(1)
        else:
            y_pred.append(0)
    print(y_pred[0:10])
    print()

    print('Resultado')
    print('CONFUSION MATRIX')
    print(confusion_matrix(labels, y_pred))
    print('Precision: ', precision_score(labels, y_pred, average='binary'))
    print('Recall: ', recall_score(labels, y_pred, average='binary'))
    print('F1 score: ', f1_score(labels, y_pred, average='binary'))
    precision_class = precision_score(labels, y_pred, average=None)
    print('Precision class/class: ', precision_class)
    recall_class = recall_score(labels, y_pred, average=None)
    print('Recall class/class: ', recall_class)
    f1_score_class = f1_score(labels, y_pred, average=None)
    print('F1 score class/class: ', f1_score_class)

def treinando_word_embeddings(corpus):
    #PASSANDO PARA LOWER CASE
    corpus_lower = []
    for sentenca in corpus:
        corpus_lower.append([word.lower() for word in sentenca])

    model = Word2Vec(min_count=1, size=100, sg=0)
    print(model)
    #criando vocabulario
    model.build_vocab(sentences=corpus_lower)
    #treinando
    model.train(sentences=corpus_lower, epochs=5, total_examples=len(corpus_lower))
    # summarize vocabulary
    words = list(model.wv.vocab)
    print(words)
    # access vector for one word
    print('EXEMPLO DE EMBEDDINGS PARA A PALAVRA ASSISTENCIA')
    print(model['assistência'])
    # save model
    #model.save('Salvo\\word_embeddings_CBOW.bin')
    # load model
    #new_model = Word2Vec.load('model.bin')
    #print(new_model)
    return model