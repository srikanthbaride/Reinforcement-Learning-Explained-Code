from ch11_policy_gradient.examples.bandit_softmax import run
if __name__ == "__main__":
    probs, theta = run(episodes=300, seed=42)
    print("Last 5 probs:\n", probs[-5:])
    print("Final probs:", probs[-1].tolist())
