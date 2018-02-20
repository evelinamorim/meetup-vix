# -*- coding: utf-8 -*-
import torch
# baseado no tutorial pytorch: http://pytorch.org/tutorials/beginner/pytorch_with_examples.html


dtype = torch.FloatTensor

N = 2 # tamanho do batch
D_in = 40 # tamanho da entrada
H = 4 # tamanho da camada escondida
D_out = 2 # tamanho da saida

# Criando entrada e saida aleatoria
x = torch.randn(N, D_in).type(dtype)
y = torch.randn(N, D_out).type(dtype)

# aleatoriamente incializando os nossos pesos
w1 = torch.randn(D_in, H).type(dtype)
w2 = torch.randn(H, D_out).type(dtype)

learning_rate = 1e-6
for t in range(500):

    # indo para frente na rede e computando a nossa classe y
    h = x.mm(w1) # multiplicacao de matriz entre x e w1
    h_relu = h.clamp(min=0) # essa funcao pega a entrada (h) e deixa entre min e max (infinito)
    # o acaba resultando na funcao relu
    y_pred = h_relu.mm(w2) # funcao relu e peso w2 para predizer a classe

    # computando a perda
    loss = (y_pred - y).pow(2).sum()
    print(t, loss)

    # similar ao codigo anterior, mas com as funcoes do torch
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    # atualizando pesos atraves da tecnica de gradiente descendente
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
