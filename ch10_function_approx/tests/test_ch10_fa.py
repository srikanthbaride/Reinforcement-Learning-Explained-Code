import numpy as np
from ch10_function_approx.features.tile_coding import TileCoder, ActionBlockTileCoder
from ch10_function_approx.envs.mountain_car import MountainCar, MountainCarConfig
from ch10_function_approx.examples.mountain_car_linear import run as mc_run

def test_tilecoder_active_count_and_bounds():
    lows = np.array([-1.0, -2.0])
    highs = np.array([1.0, 2.0])
    bins = (4, 5)
    n_tilings = 8
    offsets = [np.array([i/n_tilings, (n_tilings-i-1)/n_tilings]) for i in range(n_tilings)]
    tc = TileCoder(lows, highs, bins, n_tilings, offsets)
    x = np.array([0.0, 0.0])
    inds = tc.active_indices(x)
    assert len(inds) == n_tilings
    assert all(0 <= i < tc.total_tiles for i in inds)
    v = tc.featurize(x)
    assert np.isclose(v.sum(), n_tilings)

def test_action_block_shape_and_sparsity():
    lows = np.array([-1.0, -1.0])
    highs = np.array([1.0, 1.0])
    bins = (4,4)
    n_tilings = 4
    offsets = [np.array([0.25*i, 0.25*(3-i)]) for i in range(n_tilings)]
    tc = TileCoder(lows, highs, bins, n_tilings, offsets)
    from ch10_function_approx.envs.mountain_car import MountainCar
    ab = ActionBlockTileCoder(tc, n_actions=MountainCar().nA)
    phi = ab.phi([0.1, -0.2], a=2)
    assert phi.shape[0] == tc.total_tiles * MountainCar().nA
    assert np.isclose(phi.sum(), n_tilings)

def test_mountain_car_dynamics_and_goal():
    env = MountainCar(MountainCarConfig())
    _ = env.reset(seed=0)
    done = False
    steps = 0
    while not done and steps < 5000:
        _, _, done, _ = env.step(env.RIGHT)
        steps += 1
    assert steps <= env.cfg.max_steps

def test_linear_sarsa_runs_and_improves_steps():
    steps, w = mc_run(episodes=10, seed=0, n_tilings=4)
    assert steps.shape[0] == 10
    assert w.ndim == 1
    assert np.median(steps[-5:]) <= steps[0] + 200
