# -*- coding: utf-8 -*-
"""
@author: Eduardo Morais

Sistema de recomendação de filmes utilizando Restricted Boltzmann Machines (RBM) 

Filmes: 
    
    A bruxa                     TERROR
    Invocação do Mal            TERROR
    O Chamado                   TERROR
    Se beber nao case           COMEDIA 
    Gente grande                COMEDIA
    American Pie                COMEDIA

"""

from rbm import RBM
import numpy as np

class User:
    def __init__(self, nome, listMovies):
        self.nome = nome
        self.listMovies = listMovies

def recommenderMovie(rbm, user):
    x = rbm.run_visible(user)
    hidden_layer = np.array(x)
    return rbm.run_hidden(hidden_layer)
    
def printRecommenders(recommender, user):
    print("-----------------------------------------------------")
    for i in range(len(user.listMovies[0])):
        if user.listMovies[0,i] == 0 and recommender[0,i] == 1:
            print("Nome: %s , Recomendação: %s" %(user.nome,filmes[i]))
    
        
filmes = ["A bruxa", "Invocação do Mal", "O Chamado", 
          "Se beber nao case", "Gente grande", "American Pie"]

userTrain = np.array([[1,1,1,0,0,0],
                      [1,0,1,0,0,0],
                      [1,1,1,0,0,0],
                      [0,0,1,1,1,1],
                      [0,0,1,1,0,1],
                      [0,0,1,1,0,1]])

rbm = RBM(num_visible=6, num_hidden=2)
rbm.train(userTrain, max_epochs=5000)
rbm.weights

users = []

users.append(User("José", np.array([[1,1,0,1,0,0]])))
users.append(User("Maria", np.array([[0,0,0,1,1,0]])))

for user in users:
    recommender = recommenderMovie(rbm, user.listMovies)
    printRecommenders(recommender, user)
    
