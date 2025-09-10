import numpy as np

def truncated_lambda_return(b, c, lam):
    """
    Finite-episode λ-return for a 3-step episode with rewards (0,0,1), γ=1.
    G^(1)=b, G^(2)=c, G^(3)=1.
    G^λ = (1-λ)(G1 + λ G2) + λ^2 G3
    """
    G1, G2, G3 = b, c, 1.0
    return (1 - lam) * (G1 + lam * G2) + (lam ** 2) * G3

def test_forward_backward_equivalence():
    a, b, c = 0.5, 0.3, 0.2
    lam = 0.5

    # Forward (truncated episodic λ-return)
    Glam = truncated_lambda_return(b, c, lam)
    forward_update = Glam - a

    # Backward: TD errors and eligibilities for s0 (γ=1)
    d0 = b - a
    d1 = c - b
    d2 = 1.0 - c

    e0 = 1.0
    e1 = lam
    e2 = lam ** 2

    backward_update = d0 * e0 + d1 * e1 + d2 * e2

    assert np.isclose(forward_update, backward_update, atol=1e-12)
