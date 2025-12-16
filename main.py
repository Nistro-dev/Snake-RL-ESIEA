from snake import *
from vue import *
import genetic
import imageio
import os

# Itérations pour lesquelles on veut sauvegarder un GIF
GIF_ITERATIONS = [100, 250, 400, 600, 750, 1000]

def record_gif(nn, gameParams, filename, max_frames=200):
    vue = SnakeVue(gameParams["height"], gameParams["width"], 64)

    frames = []
    game = Game(gameParams["height"], gameParams["width"])

    while game.enCours and len(frames) < max_frames:
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
        imageio.mimsave(filename, frames, fps=10)

    return game.score


if __name__ == '__main__':
    gameParams = {"nbGames": 10, "height": 10, "width": 10}

    os.makedirs("gifs", exist_ok=True)

    def on_iteration(iteration, best_nn):
        if iteration in GIF_ITERATIONS:
            best_nn.save(f"model_iter_{iteration}.txt")
            record_gif(best_nn, gameParams, f"gifs/snake_iter_{iteration}.gif")

    nn = genetic.optimize(
        taillePopulation=400,
        tailleSelection=50,
        pc=0.8,
        mr=2.0,
        arch=[nbFeatures, 24, nbActions],
        gameParams=gameParams,
        nbIterations=1000,
        on_iteration_callback=on_iteration
    )
    nn.save("model.txt")

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
