#VARIABLES#
import numpy as np

class Reseau2neurone :
    def __init__(self, nb_couche, neurones_couche, valeur_poids_init, taille_image): #vecteur
        self.nb_couche = nb_couche
        self.neurones_couche = neurones_couche
        self.valeur_poids_init = valeur_poids_init
        self.reseau = {}
        self.taille_image = taille_image
        self.activation={} #dictionnaire des valeurs prises par chaque neurone de chaque couche
        for couche in range(self.nb_couche):
            self.reseau[couche] = np.full((neurones_couche[couche]+1,neurones_couche[couche+1]),valeur_poids_init)
            self.reseau[couche][neurones_couche[couche],:]=1
    def forward_propagation(self,image):
        if self.taille_image[0]*self.taille_image[1]!=self.neurones_couche[0]: #la taille de l'image consiste en une liste avec pour première coordonnée
            #la hauteur, et deuxieme coordonnée la largeur. Ainsi, on va avoir par exemple pour une taille de 4*4
            # les parties suivantes de l'image : 0,0;0,1;1,0;1,1. Et il faut que chaque pixel soit étudié par un neurone de la premiere couche
            raise ValueError("la taille de l'image doit correspondre au nb de neurones")
        self.activation[0]=image #l'image sera au préalable découpée par pixel, sous forme de liste, à laquelle on ajoute le biais
        #on fait ensuite produit matriciel pour avoir la valeur de la combinaison linéaire pondérée:
        for couche in range(self.nb_couche):
            activation_avec_biais=np.append(self.activation[couche],1) #on veut prendre en compte le biais donc on crée un vecteur des valeurs précédentes en y ajoutant le biais
            self.activation[couche+1]=np.dot(activation_avec_biais,self.reseau[couche]) #on fait le produit matriciel
            self.activation[couche+1]=np.where(self.activation[couche+1]>0,1,-1) #dans le dictionnaire des activations on remplace par 1 les neurones à valeur>0, -1 sinon

    def backward_propagation(self):
        pass

#nb de neurones par couche. NB couche  nb neurone. Prend en entrée matrice numpy avec taille paramétrable.
#reseau avec couche d'entrée de la taille de l'image. Paramètre blablabla puis sortie
#fonction d'activation --> si valeur du neurone supérieure à 0 alors renvoie 1, sinon on renvoie -1
#on prend tous neurones d'entrée et on multiplie par les poids, on obtiens veceur de resultat pour couche en cours,
#on le place dans fonction d'activation, puis on recommence à avancer avec les valeurs données par les focntions d'activation
#paramètres : taille de l'image
