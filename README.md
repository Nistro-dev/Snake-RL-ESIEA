# Snake

## Démonstration

### Évolution de l'apprentissage

| Itération 100 | Itération 250 | Itération 400 |
|:-------------:|:-------------:|:-------------:|
| ![Snake iter 100](gifs/snake_iter_100.gif) | ![Snake iter 250](gifs/snake_iter_250.gif) | ![Snake iter 400](gifs/snake_iter_400.gif) |

| Itération 600 | Itération 750 | Itération 1000 |
|:-------------:|:-------------:|:--------------:|
| ![Snake iter 600](gifs/snake_iter_600.gif) | ![Snake iter 750](gifs/snake_iter_750.gif) | ![Snake iter 1000](gifs/snake_iter_1000.gif) |

### Résultat final

![Snake final](gifs/snake_final.gif)

### Graphiques d'évolution

| Évolution des scores | Évolution de la longueur |
|:--------------------:|:------------------------:|
| ![Scores](graphiques/scores_evolution.png) | ![Longueur](graphiques/longueur_evolution.png) |

## Architecture du projet

```
├── main.py           # Point d'entrée principal
├── snake.py          # Moteur de jeu Snake
├── genetic.py        # Algorithme génétique
├── NN_numpy.py       # Implémentation du réseau de neurones
├── vue.py            # Interface graphique (Pygame)
├── snake.png         # Sprites du serpent
├── requirements.txt  # Dépendances Python
├── models/           # Modèles entraînés
│   └── model.txt     # Meilleur modèle
├── graphiques/       # Graphiques d'évolution
│   ├── scores_evolution.png
│   └── longueur_evolution.png
└── gifs/             # GIFs générés pendant l'entraînement
```

## Algorithme Génétique

L'algorithme génétique suit le processus suivant :

1. **Initialisation** : Création d'une population de 400 individus (réseaux de neurones) avec des poids aléatoires.

2. **Évaluation** : Chaque individu joue 10 parties de Snake. Son score est calculé selon la formule :
   ```
   score = (1 / (N × H × W × 1000)) × Σ(1000 × pommes - pas)
   ```
   Les pas sont soustraits pour récompenser les serpents rapides qui atteignent la pomme efficacement.

3. **Sélection** : Les 50 meilleurs individus sont conservés.

4. **Croisement** : Les individus sélectionnés sont croisés pour produire de nouveaux enfants. Les poids sont mélangés selon un coefficient α aléatoire.

5. **Mutation** : Des perturbations aléatoires sont appliquées aux poids et biais des enfants.

6. **Itération** : Le processus est répété pendant 1000 générations.

## Features (Entrées du réseau de neurones)

Le serpent perçoit son environnement à travers **12 features** :

### Vision à 8 directions (Raycasting)

Pour chaque direction (haut, bas, gauche, droite + 4 diagonales), un "rayon" est lancé depuis la tête du serpent. La distance jusqu'à l'obstacle (mur ou corps) est mesurée et normalisée avec la formule `1 / distance` :

- **Distance = 1** (obstacle collé) → valeur = 1.0 (danger maximal)
- **Distance = 10** (obstacle loin) → valeur = 0.1 (pas de danger)

Cette approche permet au serpent d'**anticiper les obstacles** et d'éviter de s'enfermer dans des impasses.

### Position de la pomme (4 entrées binaires)

- Pomme à gauche (0 ou 1)
- Pomme à droite (0 ou 1)
- Pomme en haut (0 ou 1)
- Pomme en bas (0 ou 1)

### Architecture du réseau

```
Entrée (12 neurones) → Couche cachée (32 neurones) → Sortie (4 neurones)
```

Les 4 sorties correspondent aux 4 directions possibles (haut, bas, gauche, droite).

## Optimisations

### Multiprocessing

L'évaluation des individus est parallélisée grâce à `concurrent.futures.ProcessPoolExecutor`. Cette optimisation permet d'exploiter tous les cœurs du processeur pour accélérer significativement l'entraînement, particulièrement lors des dernières itérations où les parties durent plus longtemps.

### Limite dynamique de pas

Pour éviter que le serpent ne tourne en rond indéfiniment, une limite de pas entre chaque pomme est imposée. Cette limite est **dynamique** et s'adapte à la taille du serpent :

```
limite = (H × W / 2) + (taille_serpent × 5)
```

- **Serpent petit (taille 4)** : limite de 70 pas → force à aller vite vers la pomme
- **Serpent grand (taille 80)** : limite de 450 pas → permet les longs détours nécessaires pour ne pas se mordre

Cette approche évite de pénaliser injustement les gros serpents qui doivent parfois faire de longs chemins pour survivre.

### Pénalité de pas (Fitness)

La fonction de fitness **soustrait** le nombre de pas au score, ce qui récompense les serpents qui atteignent la pomme rapidement. Deux serpents mangeant le même nombre de pommes seront départagés par leur efficacité de déplacement.

### Injection d'individus aléatoires

Pour éviter la stagnation (où le meilleur score ne progresse plus), **10 individus totalement aléatoires** sont injectés à chaque génération. Cela maintient la diversité génétique et permet de découvrir de nouvelles stratégies.

## Installation

### Prérequis

- Python 3.10+
- pip

### Installation des dépendances

```bash
pip install -r requirements.txt
```

### Dépendances principales

- `numpy` : Calculs matriciels pour le réseau de neurones
- `pygame` : Interface graphique du jeu
- `matplotlib` : Génération des graphiques
- `imageio` : Création des GIFs

## Utilisation

### Lancer l'entraînement

```bash
python main.py
```

L'entraînement génère automatiquement :
- Des GIFs à intervalles réguliers dans `gifs/` (itérations 100, 250, 400, 600, 750, 1000)
- Les graphiques d'évolution dans `graphiques/`
- Les modèles dans `models/`

### Mode debug

```bash
python main.py --debug
```

Le mode debug permet de visualiser le comportement du serpent aux itérations 100, 200, 300, ... 900 :
- Après chaque partie, le jeu pause et affiche : score, raison de mort, nombre de pas
- **ENTRÉE** = lancer la partie suivante
- **'q' + ENTRÉE** = quitter le mode debug et continuer l'entraînement
- Raisons de mort possibles :
  - `collision_mur` : le serpent a touché un mur
  - `collision_serpent` : le serpent s'est mordu
  - `limite_pas (X/Y)` : trop de pas sans manger de pomme
  - `victoire` : grille entièrement remplie

### Paramètres modifiables

Dans `main.py`, vous pouvez ajuster :

```python
taillePopulation=400,    # Nombre d'individus
tailleSelection=50,      # Nombre de survivants par génération
pc=0.8,                  # Probabilité de croisement
mr=2.0,                  # Taux de mutation
nbIterations=1000        # Nombre de générations
```

## Versions

### Branche [`main`](https://github.com/Nistro-dev/Snake-RL-ESIEA/tree/main) (cette branche)

Version améliorée avec :
- 12 features (vision à 8 directions + position pomme)
- Multiprocessing pour l'évaluation parallèle
- Architecture réseau : 12 → 32 → 4

### Branche [`v1`](https://github.com/Nistro-dev/Snake-RL-ESIEA/tree/v1)

Version conforme aux consignes du projet :
- 8 features (obstacles adjacents + direction + distance mur)
- Évaluation séquentielle (sans multiprocessing)
- Architecture réseau : 8 → 24 → 4

## Modèles entraînés

Le dossier `models/` contient les modèles entraînés. Pour charger un modèle existant, utilisez la méthode `load()` de la classe `NeuralNet`.

## Auteur

**Mael MICHAUD**
