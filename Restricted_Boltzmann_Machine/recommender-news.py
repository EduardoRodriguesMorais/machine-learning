# -*- coding: utf-8 -*-
"""
@author: Eduardo Morais

Sistema de recomendação de noticias utilizando Restricted Boltzmann Machines (RBM) 

Noticias: 
    Sem reforma, déficit das previdências estaduais em 2060 deve ser 4 vezes maior que o de 2013, aponta estudo     POLITICA
    Relator da reforma da Previdência se reúne com Maia e líderes para debater parecer                              POLITICA
    Lava Jato: 8 parlamentares esperam STF decidir se viram réus                                                    POLITICA
    
    Revoltado com punição, Vettel reclama muito e coloca placa de 2º lugar à frente de carro de Hamilton            ESPORTE        
    Brasil goleia Honduras por 7 a 0 na maior vitória sob o comando de Tite                                         ESPORTE
    Portugal bate Holanda e se sagra campeão da Liga das Nações                                                     ESPORTE
    
"""

from rbm import RBM
import numpy as np

class User:
    def __init__(self, nome, listNews):
        self.nome = nome
        self.listNews = listNews

def recommenderNews(rbm, user):
    x = rbm.run_visible(user)
    hidden_layer = np.array(x)
    return rbm.run_hidden(hidden_layer)
    
def printRecommenders(recommender, news, user):
    print("-----------------------------------------------------")
    for i in range(len(user.listNews[0])):
        if user.listNews[0,i] == 0 and recommender[0,i] == 1:
            print("Nome: %s , Recomendação: %s" %(user.nome,news[i]))



news = ["Sem reforma, déficit das previdências estaduais em 2060 deve ser 4 vezes maior que o de 2013, aponta estudo", "Relator da reforma da Previdência se reúne com Maia e líderes para debater parecer", "Lava Jato: 8 parlamentares esperam STF decidir se viram réus", 
          "Revoltado com punição, Vettel reclama muito e coloca placa de 2º lugar à frente de carro de Hamilton", "Brasil goleia Honduras por 7 a 0 na maior vitória sob o comando de Tite", "Portugal bate Holanda e se sagra campeão da Liga das Nações"]

userTrain = np.array([[1,1,1,0,0,0],
                      [1,0,1,0,0,0],
                      [1,1,1,0,0,0],
                      [0,0,1,1,1,1],
                      [0,0,1,1,0,1],
                      [0,0,1,1,0,1]])

users = []
users.append(User("José", np.array([[1,1,0,1,0,0]])))
users.append(User("Maria", np.array([[0,0,0,1,1,0]])))

rbm = RBM(num_visible=6, num_hidden=2)
rbm.train(userTrain, max_epochs=5000)
rbm.weights

for user in users:
    recommender = recommenderNews(rbm, user.listNews)
    printRecommenders(recommender, news, user)