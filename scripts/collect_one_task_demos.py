import argparse

import glob
import joblib
import numpy as np
import tensorflow as tf

from rllab.misc.console import query_yes_no
from rllab.sampler.utils import rollout
import pickle

files = glob.glob('data/s3/rllab-fixed-push-experts/*/*itr_300*')
files.sort()
demos_per_expert = 500
#output_dir = 'data/expert_demos/'

use_filter = True
filter_thresh = -32
joint_thresh=0.7
max_num_tries= demos_per_expert * 2
output_dir = 'data/one_task_demos/'

expert_i = 201
expert = files[expert_i]

print(expert)


with tf.Session() as sess:
    data = joblib.load(expert)
    policy = data['policy']
    env = data['env']
    xml_file = env._wrapped_env._wrapped_env.FILE
    import pdb; pdb.set_trace()
    returns = []
    demoX = []
    demoU = []
    if not use_filter:
        for _ in range(demos_per_expert):
            path = rollout(env, policy, max_path_length=100, speedup=1,
                            animated=False, always_return_paths=True)
            returns.append(path['rewards'].sum())
            demoX.append(path['observations'])
            demoU.append(path['actions'])
    else:
        num_tries = 0
        while len(returns) < demos_per_expert and num_tries < max_num_tries:
            num_tries += 1
            path = rollout(env, policy, max_path_length=100, speedup=1,
                            animated=False, always_return_paths=True)
            if path['rewards'].sum() > filter_thresh and path['observations'][-1,0] < joint_thresh:
                returns.append(path['rewards'].sum())
                demoX.append(path['observations'])
                demoU.append(path['actions'])
    if len(returns) >= demos_per_expert:
        demoX = np.array(demoX)
        demoU = np.array(demoU)
        with open(output_dir + str(expert_i) + '.pkl', 'wb') as f:
            pickle.dump({'demoX': demoX, 'demoU': demoU, 'xml':xml_file}, f, protocol=2)
tf.reset_default_graph()


