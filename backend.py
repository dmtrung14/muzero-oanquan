from muzero import MuZero
import ray

if __name__ == "__main__":

    print("\nWelcome to MuZero O An Quan!")
    muzero = MuZero("oanquan")
    # TODO: change num_tests to 100 upon deployment
    muzero.test(render=False, opponent="cross_play", num_tests=10, muzero_player=0, cross=True)
    ray.shutdown()
