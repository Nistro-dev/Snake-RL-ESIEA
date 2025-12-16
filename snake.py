import random
import itertools
import numpy
from NN_numpy import *

# tailles de couches des réseaux de neurones
nbFeatures = 12
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
        self.death_reason = None # raison de la mort (DEBUG)

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
            # Déterminer si c'est une collision mur ou serpent
            x, y = nextStep
            if x < 0 or x >= self.largeur or y < 0 or y >= self.hauteur:
                self.death_reason = "collision_mur"
            else:
                self.death_reason = "collision_serpent"
            return
        self.accessibles.remove(nextStep) #on enlève la position de la nouvelle case des positions accessibles
        if self.grille[nextStep[0]][nextStep[1]]==2: #si on mange la pomme
            self.steps = 0 #on actualise les pas et le score
            self.score+=1
            if not self.setFruit(): #s'il n'est pas possible de placer un nouveau fruit, c'est fini
                self.enCours = False
                self.death_reason = "victoire"
                return
        else:
            self.steps+=1 #comme on n'a pas mangé, on incrémente le nombre de pas
            # limite dynamique : base + bonus par segment du serpent
            limit = (self.hauteur * self.largeur // 2) + (len(self.serpent) * 5)
            if self.steps > limit: #si le nombre de pas est trop grand, c'est que l'on cycle sans manger, donc on arrête
                self.enCours = False
                self.death_reason = f"limite_pas ({self.steps}/{limit})"
                return
            self.grille[self.serpent[-1][0]][self.serpent[-1][1]] = 0 #on enlève la dernière case du serpent
            self.accessibles.append(self.serpent[-1][:])
            self.serpent = self.serpent[:-1]

        self.grille[nextStep[0]][nextStep[1]] = 1 #on ajoute la nouvelle tête
        self.serpent = [nextStep]+self.serpent

    def getFeatures(self):
        head_x, head_y = self.serpent[0]

        directions = [
            (0, -1),   # haut
            (0, 1),    # bas
            (-1, 0),   # gauche
            (1, 0),    # droite
            (-1, -1),  # diag haut-gauche
            (1, -1),   # diag haut-droite
            (-1, 1),   # diag bas-gauche
            (1, 1)     # diag bas-droite
        ]

        vision_outputs = []

        for dx, dy in directions:
            distance = 0
            x, y = head_x, head_y
            found_obstacle = False

            x += dx
            y += dy
            distance += 1

            while not found_obstacle:
                # mur
                if x < 0 or x >= self.largeur or y < 0 or y >= self.hauteur:
                    found_obstacle = True
                # corps du serpent
                elif [x, y] in self.serpent:
                    found_obstacle = True
                else:
                    x += dx
                    y += dy
                    distance += 1

            vision_outputs.append(1.0 / distance)

        # 4 entrées pour la position de la pomme
        apple_x, apple_y = self.fruit
        pomme_inputs = [
            1.0 if apple_x < head_x else 0.0,  # Pomme à gauche
            1.0 if apple_x > head_x else 0.0,  # Pomme à droite
            1.0 if apple_y < head_y else 0.0,  # Pomme en haut
            1.0 if apple_y > head_y else 0.0   # Pomme en Bas
        ]

        return numpy.array(vision_outputs + pomme_inputs)

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

