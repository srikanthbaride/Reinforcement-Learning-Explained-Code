import numpy as np

def lambda_return(b, c, lam):
    # Episode: r1=0,r2=0,r3=1, gamma=1; V(s0)=a, V(s1)=b, V(s2)=c (a cancels in diff)
    G1, G2, G3 = b, c, 1.0
    return (1 - lam) * (G1 + lam * G2 + lam**2 * G3)

def test_forward_backward_equivalence():
    a, b, c = 0.3, 0.6, 0.2
    lam = 0.5
    # forward:
    Glam = lambda_return(b, c, lam)
    forward_update = (Glam - a)

    # backward:
    d0 = b - a
    d1 = c - b
    d2 = 1.0 - c
    # eligibilities for s0 at t=0,1,2 (gamma=1)
    e0, e1, e2 = 1.0, lam, lam**2
    backward_update = d0*e0 + d1*e1 + d2*e2

    assert np.isclose(forward_update, backward_update, atol=1e-12)
