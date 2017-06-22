from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import FiniteDifferenceHvp
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.envs.mujoco.pusher_env import PusherEnv

#from gym.envs.mujoco.pusher import PusherEnv
from rllab.envs.gym_env import GymEnv
from gym.envs.mujoco import mujoco_env

from examples.autogen_cad.mujoco_render import pusher

import glob
import random

def run_task(*_):
    import pdb; pdb.set_trace()

    objs = glob.glob('/home/cfinn/code/rllab/vendor/mujoco_models/King.stl')
    random.seed(5)

    random_obj = random.choice(objs)
    random_scale = 1.0 #random.uniform(0.5, 1.0)
    random_mass = random.uniform(0.1, 2.0)
    random_damp = random.uniform(0.2, 5.0)
    # Log experiment info
    exp_log_info = {'obj': random_obj, 'scale': random_scale, 'mass': random_mass, 'damp': random_damp}

    pusher_model = pusher(mesh_file=random_obj, obj_scale=random_scale,obj_mass=random_mass,obj_damping=random_damp)
    xml_filepath = '/tmp/pusher.xml'  # Put it in exp dir, not here.
    pusher_model.save(xml_filepath)

    gym_env = GymEnv('Pusher-v0', force_reset=True, record_video=False)
    # TODO - this is hacky...
    mujoco_env.MujocoEnv.__init__(gym_env.env.env.env, xml_filepath, 5)
    import pdb; pdb.set_trace()
    env = TfEnv(normalize(gym_env))

    #env = TfEnv(normalize(GymEnv("Pusher-v0", force_reset=True, record_video=False)))
    #env = TfEnv(normalize(PusherEnv()))

    policy = GaussianMLPPolicy(
        name="policy",
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(128, 128)
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=100*500,
        max_path_length=100,
        n_itr=200,
        discount=0.99,
        step_size=0.01,
        force_batch_sampler=True,
        # optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))
        exp_log_info=exp_log_info,
    )
    algo.train()

run_experiment_lite(
    run_task,
    # Number of parallel workers for sampling
    n_parallel=8,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="gap",
    snapshot_gap=50,
    exp_prefix='trpo_push_king',
    python_command='python3',
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    seed=1,
    # plot=True,
)
