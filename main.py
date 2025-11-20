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

#nb de neurones par couche. NB couche  nb neurone. Prend en entrée matrice numpy avec taille paramétrable.
#reseau avec couche d'entrée de la taille de l'image. Paramètre blablabla puis sortie
#fonction d'activation --> si valeur du neurone supérieure à 0 alors renvoie 1, sinon on renvoie -1
#on prend tous neurones d'entrée et on multiplie par les poids, on obtiens veceur de resultat pour couche en cours,
#on le place dans fonction d'activation, puis on recommence à avancer avec les valeurs données par les focntions d'activation
#paramètres : taille de l'image
