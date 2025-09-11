import numpy as np

def run_chain(alpha=0.5, gamma=1.0, k=0, seed=None):
    """
    Two-step chain s1->s2->g, only action Right (a=0).
    One real episode then k *deterministic* planning backups on s1 (to match text/table).
    Returns (Q(s1,R), Q(s2,R)).
    """
    Q = np.zeros((3, 1))
    s1, s2, g = 0, 1, 2
    aR = 0

    # --- real episode ---
    # Step 1: s1 -> s2, r = 0
    target = 0 + gamma * np.max(Q[s2])
    Q[s1, aR] += alpha * (target - Q[s1, aR])

    # Step 2: s2 -> g (terminal), r = 1
    target = 1.0
    Q[s2, aR] += alpha * (target - Q[s2, aR])

    # --- planning backups (deterministic: always back up s1) ---
    for _ in range(k):
        target = 0 + gamma * np.max(Q[s2])
        Q[s1, aR] += alpha * (target - Q[s1, aR])

    return float(Q[s1, aR]), float(Q[s2, aR])

if __name__ == "__main__":
    for k in [0, 1, 2]:
        q1, q2 = run_chain(k=k)
        print(f"k={k}: Q(s1,R)={q1:.3f}, Q(s2,R)={q2:.3f}")
    print("Expected: k=0 -> (0.00, 0.50); k=1 -> (0.25, 0.50); k=2 -> (0.375, 0.50)")
