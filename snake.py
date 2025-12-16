import random
import itertools
import numpy
from NN_numpy import *

#Ne pas toucher, cela permet de définir les tailles de couches des réseaux de neurones
nbFeatures = 8
nbActions = 4

class Game:
    def __init__(self, hauteur, largeur):
        self.grille = [[0]*hauteur  for _ in range(largeur)] #la grille sous forme numérique : 0, 1 et 2 respectivement vide, serpent et fruit
        self.hauteur, self.largeur = hauteur, largeur #les dimensions de la grille
        self.serpent = [[largeur//2-i-1, hauteur//2] for i in range(4)] #la liste des coordonnées du serpent dans la grille
        for (x,y) in self.serpent: self.grille[x][y] = 1 
        self.direction = 3 #la direction actuelle du serpent :  0, 1, 2, 3 resp. haut, bas, gauche et droite
        self.accessibles = [[x,y] for (x,y) in list(itertools.product(range(largeur), range(hauteur))) if [x,y] not in self.serpent] #la liste des positions accessibles
        self.fruit = [0,0] #la position du fruit
        self.setFruit()
        self.enCours = True #Flag permettant de savoir si la partie est finie ou non
        self.steps = 0 #le nombre de pas effectués depuis avoir mangé
        self.score = 4 #le score actuel, défini par la taille du serpent
    
    def setFruit(self):
        if (len(self.accessibles)==0): return False #la grille est pleine, la partie est finie
        self.fruit = self.accessibles[random.randint(0, len(self.accessibles)-1)][:] #on choisit une position accessible au hasard
        self.grille[self.fruit[0]][self.fruit[1]] = 2 #on actualise la grille
        return True

    def refresh(self):
        nextStep = self.serpent[0][:] #on copie la position de la tête
        match self.direction: #on procède à un décalage d'une case, en fonction de la direction actuelle
            case 0: nextStep[1]-=1
            case 1: nextStep[1]+=1
            case 2: nextStep[0]-=1
            case 3: nextStep[0]+=1

        if nextStep not in self.accessibles: #si la nouvelle case n'est pas accessible, c'est fini
            self.enCours = False
            return
        self.accessibles.remove(nextStep) #on enlève la position de la nouvelle case des positions accessibles
        if self.grille[nextStep[0]][nextStep[1]]==2: #si on mange la pomme
            self.steps = 0 #on actualise les pas et le score
            self.score+=1
            if not self.setFruit(): #s'il n'est pas possible de placer un nouveau fruit, c'est fini
                self.enCours = False
                return
        else:
            self.steps+=1 #comme on n'a pas mangé, on incrémente le nombre de pas
            if self.steps>self.hauteur*self.largeur: #si le nombre de pas est trop grand, c'est que l'on cycle sans manger, donc on arrête
                self.enCours = False
                return
            self.grille[self.serpent[-1][0]][self.serpent[-1][1]] = 0 #on enlève la dernière case du serpent
            self.accessibles.append(self.serpent[-1][:])
            self.serpent = self.serpent[:-1]

        self.grille[nextStep[0]][nextStep[1]] = 1 #on ajoute la nouvelle tête
        self.serpent = [nextStep]+self.serpent

    def getFeatures(self):
        features = numpy.zeros(8)

        tete_x, tete_y = self.serpent[0]

        if tete_y == 0 or [tete_x, tete_y - 1] not in self.accessibles:
            features[0] = 1

        if tete_y == self.hauteur - 1 or [tete_x, tete_y + 1] not in self.accessibles:
            features[1] = 1

        if tete_x == 0 or [tete_x - 1, tete_y] not in self.accessibles:
            features[2] = 1

        if tete_x == self.largeur - 1 or [tete_x + 1, tete_y] not in self.accessibles:
            features[3] = 1

        if self.fruit[1] < tete_y:
            features[4] = 1
        elif self.fruit[1] > tete_y:
            features[4] = -1

        if self.fruit[0] > tete_x:
            features[5] = 1
        elif self.fruit[0] < tete_x:
            features[5] = -1

        features[6] = self.direction

        match self.direction:
            case 0:
                features[7] = tete_y
            case 1:
                features[7] = self.hauteur - 1 - tete_y
            case 2:
                features[7] = tete_x
            case 3:
                features[7] = self.largeur - 1 - tete_x

        return features

    def print(self):
        print("".join(["="]*(self.largeur+2)))
        for ligne in range(self.hauteur):
            chaine = ["="]
            for colonne in range(self.largeur):
                if self.grille[colonne][ligne]==1: chaine.append("#")
                elif self.grille[colonne][ligne]==2: chaine.append("F")
                else: chaine.append(" ")
            chaine.append("=")
            print("".join(chaine))
        print("".join(["="]*(self.largeur+2))+"\n")

