import numpy as np
import random
import math

class Layer:
    #Constructeur : produit une couche aléatoire
    def __init__(self, size, previousSize): 
        #Section 2 : valeurs calculés par les neurones, on stocke l'aggregation pour éviter de la recalculer lors de la retropropagation
        self.aggregations = np.zeros(size, dtype="float64")        
        self.outputs = np.zeros(size, dtype="float64")
        self.size = size

        #paramètres de la couche
        self.bias = np.array([random.uniform(-0.3, 0.3) for _ in range(size)], dtype="float64") #un biais par neurone
        self.weights = np.array([np.array([random.uniform(-1.0/np.sqrt(previousSize), 1.0/np.sqrt(previousSize)) for _ in range(previousSize)], dtype="float64") for _ in range(size)]) #pour chaque neurone, la liste des poids des arcs entrants (provenant de la couche précédente)
        
    
    #Prend en compte les valeurs de la couche précédente, y applique les poids et les biais pour calculer l'aggrégation, puis procède à l'application de la sigmoide.
    def compute(self, previousValues):
        self.aggregations = (previousValues*self.weights).sum(axis=1)+self.bias
        self.outputs = np.exp(-1*self.aggregations)
        self.outputs = 1.0/(1.0 + self.outputs)


class NeuralNet:
    #Constructeur, créer chaque couche du réseau en fonction des tailles passées en paramètres, les poids et les biais seront aléatoires
    def __init__(self, layerSizes):
        self.layers = [Layer(layerSizes[i], layerSizes[i-1] if i>=1 else 0) for i in range(len(layerSizes))]
    
    def getVector(self):
        res = np.array([])
        for layer in self.layers[1:]:
            res = np.concatenate((res, layer.bias))
            res = np.concatenate((res, layer.weights.flatten()))
        return res

    #prend une liste de features et renvoie le vecteur de sortie (un réel par classe)
    def compute(self, features):
        self.layers[0].outputs = features
        for i in range(1, len(self.layers)):
            self.layers[i].compute(self.layers[i-1].outputs)        
        return self.layers[-1].outputs

    def load(self, filename):
        with open(filename, "r") as file:
            lines = file.readlines()
            file.close()
            layerSizes = [int(size) for size in lines[0][:-1].split(" ")]
            self.layers = [Layer(layerSizes[i], layerSizes[i-1] if i>=1 else 0) for i in range(len(layerSizes))]
            idx = 1
            for layer in self.layers[1:]:
                layer.bias = np.array([float(size) for size in lines[idx][:-1].split(" ")])
                idx+=1
                for i in range(layer.size):
                    layer.weights[i] = np.array([float(size) for size in lines[idx][:-1].split(" ")])
                    idx+=1
        
    def save(self, filename):
        with open(filename, "w") as file:
            file.write(" ".join([str(layer.size) for layer in self.layers])+"\n")
            for layer in self.layers[1:]:
                file.write(" ".join([str(bias) for bias in layer.bias])+"\n")
                for weights in layer.weights:
                    file.write(" ".join([str(weight) for weight in weights])+"\n")
            file.close()