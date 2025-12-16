import numpy
import matplotlib.pyplot as plt
from NN_numpy import *
from snake import *

# Historique pour le graphique
history_best = []
history_avg = []
history_longueur = []

def eval(sol, gameParams):
    N = gameParams["nbGames"]
    H = gameParams["height"]
    W = gameParams["width"]

    total = 0.0
    max_longueur = 0

    for _ in range(N):
        # init partie
        game = Game(H, W)

        while game.enCours:
            features = game.getFeatures()

            prediction = sol.nn.compute(features)

            direction = numpy.argmax(prediction)
            game.direction = direction

            game.refresh()

        # calcul du score
        longueur = game.score  # taille du serpent
        if longueur > max_longueur:
            max_longueur = longueur
        pas = game.steps
        total += 1000 * (longueur - 4) + pas

    # score final
    sol.score = total / (N * H * W * 1000)
    sol.longueur = max_longueur  # longueur max atteinte

    return (sol.id, sol.score)

class Individu:
    def __init__(self, nn, id):
        self.nn = nn
        self.id = id
        self.score = 0
        self.longueur = 0

    def clone(self, copie):
        for idx, layer in enumerate(copie.nn.layers[1:]):
            layer.bias = self.nn.layers[idx+1].bias.copy()
            layer.weights = self.nn.layers[idx+1].weights.copy()


def croisement(parent1, parent2, enfant1, enfant2, pc):
    if numpy.random.random() > pc:
        parent1.clone(enfant1)
        parent2.clone(enfant2)
        return

    for idx in range(1, len(parent1.nn.layers)):
        # alpha entre 0 et 1
        alpha = numpy.random.random()

        layer_p1 = parent1.nn.layers[idx]
        layer_p2 = parent2.nn.layers[idx]
        layer_e1 = enfant1.nn.layers[idx]
        layer_e2 = enfant2.nn.layers[idx]

        # melanger les poids
        layer_e1.weights = alpha * layer_p1.weights + (1 - alpha) * layer_p2.weights
        layer_e2.weights = (1 - alpha) * layer_p1.weights + alpha * layer_p2.weights

        # melanger les biais
        layer_e1.bias = alpha * layer_p1.bias + (1 - alpha) * layer_p2.bias
        layer_e2.bias = (1 - alpha) * layer_p1.bias + alpha * layer_p2.bias


def mutation(individu, mr):
    for idx in range(1, len(individu.nn.layers)):
        layer = individu.nn.layers[idx]
        previousLayer = individu.nn.layers[idx - 1]

        # proba de mutation
        pm_biais = mr / layer.size
        pm_poids = mr / previousLayer.size

        # mutationn biais
        for j in range(layer.size):
            if numpy.random.random() < pm_biais:
                layer.bias[j] += numpy.random.randn()

        # mutation poids
        for j in range(layer.size):
            for i in range(previousLayer.size):
                if numpy.random.random() < pm_poids:
                    layer.weights[j][i] += numpy.random.randn()


def optimize(taillePopulation, tailleSelection, pc, mr, arch, gameParams, nbIterations, on_iteration_callback=None):
    P = taillePopulation
    S = tailleSelection

    # pop initiale
    population = []
    for i in range(P):
        nn = NeuralNet(arch)
        individu = Individu(nn, i)
        population.append(individu)

    for individu in population:
        eval(individu, gameParams)

    # boucle principale
    for iteration in range(nbIterations):

        population.sort(key=lambda ind: ind.score, reverse=True)
        population = population[:S]  # garde les meilleures

        # calcul stats
        best_score = population[0].score
        avg_score = sum(ind.score for ind in population) / len(population)
        history_best.append(best_score)
        history_avg.append(avg_score)

        best_longueur = population[0].longueur
        history_longueur.append(best_longueur)
        avg_longueur = sum(history_longueur) / len(history_longueur)

        print(f"Iteration {iteration + 1}/{nbIterations} - Best: {best_score:.4f} - Avg: {avg_score:.4f} - Longueur (best): {best_longueur} - Longueur (avg): {avg_longueur:.1f}")

        if on_iteration_callback is not None:
            on_iteration_callback(iteration + 1, population[0].nn)

        # création nouveaux individu
        nouveaux = []
        while len(nouveaux) < P - S:
            # selection 2 parents
            parent1 = population[numpy.random.randint(0, S)]
            parent2 = population[numpy.random.randint(0, S)]

            # creation 2 enfants
            enfant1 = Individu(NeuralNet(arch), len(population) + len(nouveaux))
            enfant2 = Individu(NeuralNet(arch), len(population) + len(nouveaux) + 1)

            # croisement
            croisement(parent1, parent2, enfant1, enfant2, pc)

            # mutation
            mutation(enfant1, mr)
            mutation(enfant2, mr)

            eval(enfant1, gameParams)
            eval(enfant2, gameParams)

            nouveaux.append(enfant1)
            if len(nouveaux) < P - S:
                nouveaux.append(enfant2)

        population.extend(nouveaux)

    # return le meilleur
    population.sort(key=lambda ind: ind.score, reverse=True)
    print(f"Best score: {population[0].score:.4f}")

    save_plot()

    return population[0].nn


def save_plot():
    import os
    os.makedirs("graphiques", exist_ok=True)

    iterations = range(1, len(history_best) + 1)

    # graphique scores
    plt.figure(figsize=(12, 6))
    plt.plot(iterations, history_best, label='best score', color='blue')
    plt.plot(iterations, history_avg, label='avg score', color='orange')
    plt.xlabel('iteration')
    plt.ylabel('score')
    plt.title('évolution des scores')
    plt.legend()
    plt.grid(True)
    plt.savefig('graphiques/scores_evolution.png', dpi=150)
    plt.close()

    # graphique longueurs
    plt.figure(figsize=(12, 6))
    plt.plot(iterations, history_longueur, label='longueur best', color='green')
    plt.xlabel('iteration')
    plt.ylabel('longueur')
    plt.title('évolution de la longueur')
    plt.legend()
    plt.grid(True)
    plt.savefig('graphiques/longueur_evolution.png', dpi=150)
    plt.close()
