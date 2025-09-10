import numpy as np

def lambda_return(b, c, lam):
    G1 = b
    G2 = c
    G3 = 1.0
    return (1 - lam) * (G1 + lam * G2 + lam**2 * G3)

def test_forward_backward_equivalence():
    a, b, c = 0.5, 0.3, 0.2
    lam = 0.5

    # forward λ-return update for V(s0) = a
    Glam = lambda_return(b, c, lam)
    forward_update = Glam - a

    # backward view TD error updates
    d0 = b - a
    d1 = c - b
    d2 = 1.0 - c

    # eligibilities for s0 at each step
    e0 = 1.0
    e1 = lam       # after one step
    e2 = lam**2    # after two steps (γ=1 here)

    backward_update = d0 * e0 + d1 * e1 + d2 * e2

    assert np.isclose(forward_update, backward_update, atol=1e-12)
