"""Step 1 - build the training and test sets with the simulator.

Training drop : random users + fading; PER_DROP random assignments are
                refined and their true power recorded (label).
Test drop     : ALL 240 assignments are refined, so the best possible
                ("oracle") choice is known for evaluation.
Usage: python s01_generate_dataset.py [n_train] [n_test]
"""
import sys
import numpy as np
from multiprocessing import Pool, cpu_count
from pass_sim import new_trial
from pass_ml import ASSIGNMENTS, features, refine_assignment, geometry

PER_DROP = 60
TRAIN_SEED0, TEST_SEED0 = 100_000, 900_000     # disjoint seed ranges


def make_trial(seed):
    return new_trial(np.random.default_rng(seed), fading=True)


def train_drop(seed):
    tr = make_trial(seed)
    idx = np.random.default_rng(seed + 7).choice(len(ASSIGNMENTS), PER_DROP, replace=False)
    y = [refine_assignment(tr, ASSIGNMENTS[i])[0] for i in idx]
    return features(tr, ASSIGNMENTS[idx]), np.array(y)


def test_drop(seed):
    tr = make_trial(seed)
    return np.array([refine_assignment(tr, a)[0] for a in ASSIGNMENTS])


if __name__ == "__main__":
    n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    n_test = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    with Pool(cpu_count()) as pool:
        tr_out = pool.map(train_drop, range(TRAIN_SEED0, TRAIN_SEED0 + n_train), chunksize=10)
        te_out = pool.map(test_drop, range(TEST_SEED0, TEST_SEED0 + n_test), chunksize=5)
    np.savez_compressed(
        "data/dataset.npz",
        X_train=np.vstack([o[0] for o in tr_out]), y_train=np.concatenate([o[1] for o in tr_out]),
        groups=np.repeat(np.arange(n_train), PER_DROP),
        test_seeds=np.arange(TEST_SEED0, TEST_SEED0 + n_test), test_power=np.array(te_out))
    print(f"saved data/dataset.npz: {n_train} training drops x {PER_DROP} assignments, {n_test} test drops x {len(ASSIGNMENTS)}")
