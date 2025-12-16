import os
import argparse
import time
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

from snake import *
from vue import *
import genetic
import imageio

# Itérations pour lesquelles on veut sauvegarder un GIF
GIF_ITERATIONS = [100, 250, 400, 600, 750, 1000]
# Itérations pour lesquelles on veut le debug
DEBUG_ITERATIONS = [100, 200, 300, 400, 500, 600, 700, 800, 900]

def record_gif(nn, gameParams, filename, target_duration=30):
    vue = SnakeVue(gameParams["height"], gameParams["width"], 64)

    frames = []
    game = Game(gameParams["height"], gameParams["width"])

    while game.enCours:
        pred = np.argmax(nn.compute(game.getFeatures()))
        game.direction = pred
        game.refresh()
        if not game.enCours:
            break
        vue.displayGame(game)
        frame = pygame.surfarray.array3d(vue.game_window)
        frame = frame.transpose([1, 0, 2])
        frames.append(frame)

    if frames:
        fps = max(20, min(60, len(frames) / target_duration))
        imageio.mimsave(filename, frames, fps=fps)

    return game.score, game.death_reason


def debug_play(nn, gameParams):
    vue = SnakeVue(gameParams["height"], gameParams["width"], 64)
    fps = pygame.time.Clock()

    print("\n" + "="*50)
    print("[DEBUG] Début visualisation.")
    print("[DEBUG] ENTRÉE = partie suivante | 'q' + ENTRÉE = quitter")
    print("="*50)

    partie_num = 0
    while True:
        partie_num += 1
        game = Game(gameParams["height"], gameParams["width"])
        step_count = 0

        while game.enCours:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return None, None

            pred = np.argmax(nn.compute(game.getFeatures()))
            game.direction = pred
            game.refresh()
            step_count += 1

            if not game.enCours:
                break
            vue.displayGame(game)
            fps.tick(15)

        print(f"[DEBUG] Partie {partie_num} - Score: {game.score} - Mort: {game.death_reason} - Steps: {step_count}")
        print("[DEBUG] ENTRÉE = partie suivante | 'q' + ENTRÉE = quitter")

        user_input = input().strip().lower()
        if user_input == 'q':
            break

    print("="*50)
    print("[DEBUG] Fin visualisation")

    return game.score, game.death_reason


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Snake IA - Algorithme Génétique')
    parser.add_argument('--debug', action='store_true', help='mode debug')
    args = parser.parse_args()

    DEBUG_MODE = args.debug
    genetic.DEBUG_MODE = DEBUG_MODE  # Passer le mode debug à genetic.py
    if DEBUG_MODE:
        print("[DEBUG] Activé")

    gameParams = {"nbGames": 10, "height": 10, "width": 10}

    os.makedirs("gifs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    def on_iteration(iteration, best_nn):
        if iteration in GIF_ITERATIONS:
            best_nn.save(f"models/model_iter_{iteration}.txt")
            record_gif(best_nn, gameParams, f"gifs/snake_iter_{iteration}.gif")

        # DEBUG: visualisation tous les 50 itérations
        if DEBUG_MODE and iteration in DEBUG_ITERATIONS:
            print(f"\n[DEBUG] === BREAKPOINT ITERATION {iteration} ===")
            debug_play(best_nn, gameParams)

    nn = genetic.optimize(
        taillePopulation=400,
        tailleSelection=50,
        pc=0.8,
        mr=2.0,
        arch=[nbFeatures, 32, nbActions],
        gameParams=gameParams,
        nbIterations=1000,
        on_iteration_callback=on_iteration
    )
    nn.save("models/model.txt")

    record_gif(nn, gameParams, "gifs/snake_final.gif")

    vue = SnakeVue(gameParams["height"], gameParams["width"], 64)
    fps = pygame.time.Clock()
    gameSpeed = 20

    while True:
        game = Game(gameParams["height"], gameParams["width"])
        while game.enCours:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
            pred = np.argmax(nn.compute(game.getFeatures()))
            game.direction = pred
            game.refresh()
            if not game.enCours:
                break
            vue.displayGame(game)
            fps.tick(gameSpeed)
