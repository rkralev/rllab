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
demos_per_expert = 8
#output_dir = 'data/expert_demos/'

#use_filter = True
#filter_thresh = -34
#joint_thresh = 1.0
#max_num_tries=16
#output_dir = 'data/expert_demos_filter_joint0/'

use_filter = True
filter_thresh = -32
joint_thresh=0.7
max_num_tries=12
output_dir = 'data/expert_demos_filter_rew32_jnt7e-1_tries12/'


for expert, expert_i in zip(files, range(len(files))):
    if expert_i % 25 == 0:
        print('collecting #' + str(expert_i))
    if '2017_06_23_21_04_45_0091' in expert:
        continue
    with tf.Session() as sess:
        data = joblib.load(expert)
        policy = data['policy']
        env = data['env']
        xml_file = env._wrapped_env._wrapped_env.FILE
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


