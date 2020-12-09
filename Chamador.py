print('Carregando..')
import Neural_net_model as nnm
import pandas as pd
from keras.models import load_model
import nltk
import pickle as pkl
import Pre_processo as pp
import numpy.core._dtype_ctypes
import sklearn

def criando_train_test():
    print('>>>>>> CARREGANDO CORPUS')
    user_file = open('nomedoarquivo.pkl', 'rb')
    context = pkl.load(user_file)
    user_file.close()
    system_file = open('nomedoarquivo.pkl', 'rb')
    utterance = pkl.load(system_file)
    system_file.close()
    labels_file = open('nomedoarquivo.pkl', 'rb')
    labels = pkl.load(labels_file)
    labels_file.close()
    print('Labels: ', numpy.unique(labels, return_counts=True))

    print('>>>>>> GERANDO DADOS DE TREINAMENTO E TESTE')
    context_train, context_test, utterance_train, utterance_test, labels_train, labels_test = sklearn.model_selection.train_test_split(
        context, utterance, labels, stratify=labels, test_size=0.2)
    print('Quantidade de treinamento: ', numpy.unique(labels_train, return_counts=True))
    print('Quantidade de test: ', numpy.unique(labels_test, return_counts=True))

    c_train = open('nomedoarquivo.pkl', 'wb')
    pkl.dump(context_train, c_train)
    c_train.close()
    u_train = open('nomedoarquivo.pkl', 'wb')
    pkl.dump(utterance_train, u_train)
    u_train.close()
    l_train = open('nomedoarquivo.pkl', 'wb')
    pkl.dump(labels_train, l_train)
    l_train.close()
    c_test = open('nomedoarquivo.pkl', 'wb')
    pkl.dump(context_test, c_test)
    c_test.close()
    u_test = open('nomedoarquivo.pkl', 'wb')
    pkl.dump(utterance_test, u_test)
    u_test.close()
    l_test = open('nomedoarquivo.pkl', 'wb')
    pkl.dump(labels_test, l_test)
    l_test.close()


def treinando_chatbot(context_train, utterance_train, labels_train):
    print('>>>>>> PRINTANDO EXEMPLOS')
    print('Usuario: ', context_train[0])
    print('Sistema: ', utterance_train[0])
    print('Label: ', labels_train[0])

    print('>>>>>> TOKENIZACAO')
    #context_train = pp.tokenizacao_por_palavra(context_train)
    #utterance_train = pp.tokenizacao_por_palavra(utterance_train)

    print('>>>>>> TREINANDO WORD EMBEDDINGS')
    word_embeddings = nnm.treinando_word_embeddings(context_train + utterance_train)

    print('>>>>>> PRE PROCESSAMENTO')
    context_train = pp.remove_stopwords(context_train)
    utterance_train = pp.remove_stopwords(utterance_train)
    context_train = pp.voltando_string(context_train)
    utterance_train = pp.voltando_string(utterance_train)

    print('>>>>>> RELIZANDO ENCODING DA LABELS')
    print('Labels: ', labels_train[0:10])
    encoder = sklearn.preprocessing.LabelEncoder()
    encoder.fit(labels_train)
    print('Labels: ', encoder.classes_)
    labels_train = encoder.transform(labels_train)
    print('Labels: ', labels_train[0:10])
    # output = open('Salvo\\encoder.pkl', 'wb')
    # pkl.dump(encoder, output)
    # output.close()

    print('>>>>>> CRIANDO VOCABULARIO')
    tokenizer = pp.vocabulario(context_train, utterance_train)

    print('>>>>>> PASSANDO SENTENCAS PARA SEQUENCIA DE NUMEROS')
    sentence_size = pp.tirando_media_sentencas(context_train + utterance_train)
    context_train = pp.gerando_sequencia_numerica(tokenizer, context_train, sentence_size)
    utterance_train = pp.gerando_sequencia_numerica(tokenizer, utterance_train, sentence_size)

    print('>>>>>> CRIANDO EMBEDDING MATRIX')
    embedding_matrix = nnm.criando_embeddings_matrix(tokenizer, word_embeddings)

    print('>>>>>> TREINANDO MODELO')
    encoder_model = nnm.encoder_context_utterance(len(tokenizer.word_index), sentence_size, embedding_matrix)
    dual_lstm = nnm.dual_lstm(encoder_model, sentence_size)
    dual_lstm.fit([context_train, utterance_train], labels_train, batch_size=32, epochs=1, verbose=1)
    #dual_lstm.save("Salvo\\dual_lstm.h5")

    print('>>>>>> TREINAMENTO CONCLUIDO')

def testando_chatbot(context_test, utterance_test, labels_test):
    print('>>>>>> PRE PROCESSAMENTO')
    # context_test = pp.tokenizacao_por_palavra(context_test)
    # utterance_test = pp.tokenizacao_por_palavra(utterance_test)
    context_test = pp.remove_stopwords(context_test)
    utterance_test = pp.remove_stopwords(utterance_test)
    context_test = pp.voltando_string(context_test)
    utterance_test = pp.voltando_string(utterance_test)

    print('>>>>>> PASSANDO SENTENCAS PARA SEQUENCIA DE NUMEROS')
    tok_file = open('Salvo\\tokenizer.pkl', 'rb')
    tokenizer = pkl.load(tok_file)
    tok_file.close()
    sent_file = open('Salvo\\media.pkl', 'rb')
    sentence_size = pkl.load(sent_file)
    sent_file.close()
    context_test = pp.gerando_sequencia_numerica(tokenizer, context_test, sentence_size)
    utterance_test = pp.gerando_sequencia_numerica(tokenizer, utterance_test, sentence_size)

    print('>>>>>> ENCODING DAS LABELS')
    enc_file = open('Salvo\\encoder.pkl', 'rb')
    encoder = pkl.load(enc_file)
    enc_file.close()
    labels_test = encoder.transform(labels_test)
    print('Labels test: ', labels_test[0:5])

    print('>>>>>> TESTANDO O MODELO')
    dual_lstm = load_model('Salvo\\dual_lstm.h5')

    nnm.test_model(dual_lstm, context_test, utterance_test, labels_test)


#criando_train_test()

print('>>>>>> CARREGANDO DADOS')
c_train = open('nomedoarquivo.pkl', 'rb')
context_train = pkl.load(c_train)
c_train.close()
u_train = open('nomedoarquivo.pkl', 'rb')
utterance_train = pkl.load(u_train)
u_train.close()
l_train = open('nomedoarquivo.pkl', 'rb')
labels_train = pkl.load(l_train)
l_train.close()

treinando_chatbot(context_train, utterance_train, labels_train)

c_test = open('nomedoarquivo.pkl', 'rb')
context_test = pkl.load(c_test)
c_test.close()
u_test = open('nomedoarquivo.pkl', 'rb')
utterance_test = pkl.load(u_test)
u_test.close()
l_test = open('nomedoarquivo.pkl', 'rb')
labels_test = pkl.load(l_test)
l_test.close()

testando_chatbot(context_test, utterance_test, labels_test)

