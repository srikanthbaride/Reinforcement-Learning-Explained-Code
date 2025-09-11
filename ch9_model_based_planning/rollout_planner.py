import numpy as np

def rollout_q_update(Q, model, s, a_seq, alpha=0.5, gamma=1.0):
    """
    Simulate a short rollout from state s using learned model.
    a_seq: sequence of action indices; update Q[s,a_seq[0]] with MC target.
    """
    s_curr = s
    G = 0.0
    g = 1.0
    for a in a_seq:
        s_next, r = model.predict(s_curr, a)
        G += g * r
        g *= gamma
        s_curr = s_next
    a0 = a_seq[0]
    Q[s, a0] += alpha * (G - Q[s, a0])
    return Q[s, a0]
