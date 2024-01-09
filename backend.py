from muzero import MuZero
import ray

if __name__ == "__main__":
    muzero = MuZero("oanquan")
    # TODO: change num_tests to 100 upon deployment
    result = muzero.test(render=False, opponent="cross_play", num_tests=10, muzero_player=0, cross=True)
    print(str(1 - result))
    ray.shutdown()
