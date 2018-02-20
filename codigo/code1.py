# -*- coding: utf-8 -*-
import numpy as np

# baseado no tutorial pytorch: http://pytorch.org/tutorials/beginner/pytorch_with_examples.html

N = 2 # tamanho do batch
D_in = 40 # tamanho da entrada
H = 4 # tamanho da camada escondida
D_out = 2 # tamanho da saida


# Criando entrada e saida aleatoria
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# aleatoriamente incializando os nossos pesos
# lembra do nosso desenho do slide?
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

# qual o passo que quero dar contra a derivada?
learning_rate = 1e-6

# isso aqui e cada epoca
for t in range(500):

    # indo para frente na rede e computando a nossa classe y
    h = x.dot(w1)
    h_relu = np.maximum(h, 0) # lembra que o relu e entre 0 e infinito e nao tem valor negativo?
    y_pred = h_relu.dot(w2) # aqui estamos usando a funcao de ativacao relu

    # aqui vamos computar a nossa perda
    loss = np.square(y_pred - y).sum()
    print(t, loss)

    # computamos a perda entao temos que calcular aqui os gradientes (derivadas)
    # para saber como voltar de acordo com a perda
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    # atualizando pesos
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
