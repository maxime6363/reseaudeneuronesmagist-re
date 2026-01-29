import numpy as np
def sigmoide(z):
    return 1/(1+np.exp(-z))

def derivee_sigmoide(z):
    return sigmoide(z)*(1-sigmoide(z))

def cout(prediction,label):
    return ((prediction-label)**2).mean()

class Reseau2neurones :
    def __init__(self,neurones,poids,biais):
        self.neurones=neurones
        self.poids=poids
        self.biais=biais
        self.activation=

    def forward_propagation(self,image):



