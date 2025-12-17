#VARIABLES#
import numpy as np
import random as rd

from ReadingMnist import MnistDataloader


def sigmoide(z):
    return 1/(1+np.exp(-z))

def derivee_sigmoide(z):
    return sigmoide(z)*(1-sigmoide(z))

class Reseau2neurone :
    def __init__(self, nb_couche, neurones_couche,taux_apprentissage): #vecteur
        self.nb_couche=nb_couche
        self.nb_neurones_couche=neurones_couche
        self.reseau_poids={}
        self.activation={}
        self.gradients={} #dictionnaire des gradients,
        self.taux_apprentissage=taux_apprentissage
        for couche in range(self.nb_couche):
            nb_neurones_entree=neurones_couche[couche]+1 #le plus 1 correspond en fait au biais
            nb_neurones_sortie=neurones_couche[couche+1]
            self.reseau_poids[couche]=np.full((nb_neurones_entree,nb_neurones_sortie),1)

    def forward_propagation(self,image):
        self.activation[0]=image #l'image sera au préalable découpée par pixel, sous forme de liste, à laquelle on ajoute le biais
        #on fait ensuite produit matriciel pour avoir la valeur de la combinaison linéaire pondérée:
        for couche in range(self.nb_couche):
            activation_avec_biais = np.append(self.activation[couche], 1)#on veut prendre en compte le biais donc on crée un vecteur des valeurs précédentes en y ajoutant le biais
            self.activation[couche + 1] = sigmoide(np.dot(activation_avec_biais,self.reseau_poids[couche]))  # on fait le produit matriciel et on active avec la fonction sigmoïde
        return self.activation[self.nb_couche]

    def cout(self,image,label): #calcule les différences entre ce qu'on recherche en sortie et ce qu'on a en sortie
        sortie=self.forward_propagation(image)
        cible=np.zeros(self.nb_neurones_couche[-1])
        cible[label]=1
        cout_total=np.sum((sortie-cible)**2)
        return cout_total

    def calcul_gradient(self,image,label):
        sortie=self.forward_propagation(image)
        cible = np.zeros(self.nb_neurones_couche[-1])
        cible[label] = 1
        erreurs={}
        nombre_matrices_poids=self.nb_couche -1
        erreurs[nombre_matrices_poids]=2*(sortie-cible)*derivee_sigmoide(sortie)
        for couche in range(nombre_matrices_poids-1,-1,-1):
            derivee=derivee_sigmoide(self.activation[couche+1])
            poids_suivants=self.reseau_poids[couche+1][:-1,:] #on ne prend pas en compte le biais
            somme=np.dot(erreurs[couche+1],poids_suivants.T)
            erreurs[couche]=derivee*somme
        for couche in range(self.nb_couche):
            activation_avec_biais=np.append(self.activation[couche], 1)
            n = len(activation_avec_biais)
            m = len(erreurs[couche])
            gradient = np.zeros((n, m))
            for i in range(n):
                for j in range(m):
                    gradient[i, j] = activation_avec_biais[i] * erreurs[couche][j]
            self.gradients[couche] = gradient
        return self.gradients

    def mettre_a_jour_poids(self):
        for couche in range(self.nb_couche):
            self.reseau_poids[couche] -= self.taux_apprentissage * self.gradients[couche]

