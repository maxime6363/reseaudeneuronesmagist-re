#VARIABLES#
import numpy as np
class Reseau2neurone :
    def __init__(self, nb_couche, neurones_couche, valeur_poids_init, taille_image): #vecteur
        self.reseau = {}
        for couche in range(0,nb_couche-1):
            self.reseau[couche] = np.full((neurones_couche[couche+1],neurones_couche[couche]+1),valeur_poids_init)
            self.reseau[couche][:,neurones_couche[couche]]=1
    def forward_propagation(self):
        pass

    def backward_propagation(self):
        pass

