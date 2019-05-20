# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 18:38:39 2019

@author: Eduardo
"""

import pandas as pd 
previsores = pd.read_csv("dataset/entradas-breast.csv")
classes = pd.read_csv("dataset/saidas-breast.csv")


from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_treste, classe_treinamento, classe_teste = train_test_split(previsores, classes, test_size=0.25)            


from keras.models import Sequential
from keras.layers import Dense

classificador = Sequential()
#Adiciona forma a novo modelo. 16 camadas de profundidade, função de ativação relu, inicialização randômica, 30 camadas de entrada. 
classificador.add(Dense(units = 16, activation = 'relu', 
                        kernel_initializer = 'random_uniform', input_dim = 30))

#Adiciona camada de saída. 1 neurônio de saída, função de ativação sigmoid 
classificador.add(Dense(units = 1, activation = 'sigmoid'))

#Configuta o modelo para treinamento. Calculo de ajuste dos pesos(Descida do Gradiente Adam), classificação binária, métrica para avaliação do modelo
classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                      metrics = ['binary_accuracy'])

#Inicia treinamento do modelo. Previsores do treinamento, Classe pro treinamento. Épocas para reajuste dos pesos, épocas de treinamento
classificador.fit(previsores_treinamento, classe_treinamento,
                  batch_size = 10, epochs = 100)

previsoes = classificador.predict(previsores_treste)
previsoes = previsoes > 0.5

#Verifica resultado do treinamento pelo SKlearn
from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes) 

#Verifica resultado do treinamento pelo Keras
resultado = classificador.evaluate(previsores_treste, classe_teste)