import argparse

import joblib
import numpy as np
import tensorflow as tf

from rllab.misc.console import query_yes_no
from rllab.sampler.utils import rollout

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--max_path_length', type=int, default=100,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=1,
                        help='Speedup')
    parser.add_argument('--animated', type=bool, default=True,
                        )
    args = parser.parse_args()

    # If the snapshot file use tensorflow, do:
    # import tensorflow as tf
    # with tf.Session():
    #     [rest of the code]
    with tf.Session() as sess:
        data = joblib.load(args.file)
        policy = data['policy']
        env = data['env']
        returns = []
        import pdb; pdb.set_trace()
        while True:
            path = rollout(env, policy, max_path_length=args.max_path_length,
                           animated=False, speedup=args.speedup, always_return_paths=True)
            print('Return: '+str(path['rewards'].sum()))
            returns.append(path['rewards'].sum())
            print('Average Return so far: ' + str(np.mean(returns)))
            if not query_yes_no('Continue simulation?'):
                break
