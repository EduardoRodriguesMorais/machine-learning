# -*- coding: utf-8 -*-
"""
    Algoritmo Bayesiano Multinominal
    Objetivo: Classificar Porco ou Cachorro
    
    
    Classificação: 
        
    [     1    ,        1          ,     1    ] 
    [é gordinho, tem perninha curta, faz auau ] 
    
    Marcações: 1 = Porco
              -1 = Cachorro  
    
"""


from sklearn.naive_bayes import MultinomialNB

porco1 = [1,1,0]
porco2 = [1,1,0]
porco3 = [1,1,0]
porco4 = [0,0,0]
cachorro1 = [0,1,1]
cachorro2 = [1,0,1]
cachorro3 = [0,0,1]
cachorro4 = [1,1,0]

dados = [porco1, porco2, porco3, porco4, cachorro1, cachorro2, cachorro3,cachorro4]
marcacoes = [1,1,1,1,-1,-1,-1,-1]        
misterioso1 = [1,0,1] #Cachorro
misterioso2 = [1,1,1] #Cachorro
misterioso3 = [1,0,0] #Porco
misterioso4 = [1,1,0] #Porco
misterioso5 = [0,0,0] #Porco
marcacoes_teste = [-1,-1,1,1,1]
teste = [misterioso1, misterioso2, misterioso3, misterioso4,misterioso5]

modelo = MultinomialNB()
modelo.fit(dados, marcacoes) #.fit Adequa dados e marcações

resultado = modelo.predict(teste)#.predict Preve quem são  
diferenca = resultado - marcacoes_teste 

print(resultado)
print(marcacoes_teste)
print(diferenca)
acertos = [d for d in diferenca if d == 0]

total_acertos = len(acertos)
    

print('Total acertos: '+str(total_acertos)+'  Acertos:'+ str(acertos))

printResult = ''
for i in range(len(marcacoes_teste)):
    if diferenca[i] == 0:
        printResult = '_Acertou_'
    else:
        printResult = ' Errou'
    
    if marcacoes_teste[i] > 0:
            print('Misterioso '+str(i+1) +' é um Porco    - Previsão: ' + printResult)
    else:
         print('Misterioso '+str(i+1) +' é um Cachorro - Previsão: ' + printResult)

        