from ch9_model_based_planning.examples.chain_example import run_chain

def test_chain_numbers_seed0():
    q1, q2 = run_chain(k=0, seed=0)
    assert abs(q1 - 0.00) < 1e-6
    assert abs(q2 - 0.50) < 1e-6
    q1, q2 = run_chain(k=1, seed=0)
    assert abs(q1 - 0.25) < 1e-6
    assert abs(q2 - 0.50) < 1e-6
    q1, q2 = run_chain(k=2, seed=0)
    assert abs(q1 - 0.375) < 1e-6
    assert abs(q2 - 0.50) < 1e-6
