from ch10_function_approx.examples.mountain_car_linear import run
if __name__ == "__main__":
    steps, w = run(episodes=30, seed=42, n_tilings=8)
    print("Steps per episode:", steps.tolist())
