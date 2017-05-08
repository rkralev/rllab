import os.path as osp
from rllab.envs.base import Env
from rllab.spaces import Box
from rllab.envs.base import Step
from rllab.misc import logger
import numpy as np
from sandbox.rocky.tf.policies.base import Policy
from rllab.misc.overrides import overrides
from rllab.core.serializable import Serializable
import matplotlib.pyplot as plt


class PointEnvRandGoal(Env):
    def __init__(self, goal=None):  # Can set goal to test adaptation.
        self._goal = goal

    @property  # this should have the same name for all: ant/cheetah direction,... and be a list of params
    def objective_params(self):
        return self._goal

    @property
    def observation_space(self):
        return Box(low=-np.inf, high=np.inf, shape=(2,))

    @property
    def action_space(self):
        return Box(low=-0.1, high=0.1, shape=(2,))

    def reset(self, objective_params=None, clean_reset=False):
        if clean_reset:
            # print("cleaning goal")
            self._goal = None
        else:
            goal = objective_params
            if goal is not None:
                # print("using given goal")
                self._goal = goal
            elif self._goal is None:
                # print("sampling a new goal")
                # Only set a new goal if this env hasn't had one defined before or if it has been cleaned
                self._goal = np.random.uniform(-0.5, 0.5, size=(2,))
                # goals = [np.array([-0.5,0]), np.array([0.5,0])]
                # goals = np.array([[-0.5,0], [0.5,0],[0.2,0.2],[-0.2,-0.2],[0.5,0.5],[0,0.5],[0,-0.5],[-0.5,-0.5],[0.5,-0.5],[-0.5,0.5]])
                # self._goal = goals[np.random.randint(10)]

        self._state = (0, 0)
        observation = np.copy(self._state)
        return observation

    def step(self, action):
        # print("inside env step: _state, goal, action:", self._state, self._goal, self. action)
        self._state = self._state + action
        x, y = self._state
        x -= self._goal[0]
        y -= self._goal[1]
        reward = - (x ** 2 + y ** 2) ** 0.5
        done = abs(x) < 0.01 and abs(y) < 0.01
        next_observation = np.copy(self._state)
        return Step(observation=next_observation, reward=reward, done=done, goal=self._goal)  # goal goes to env_infos

    def render(self):
        print('current state:', self._state)

    def log_diagnostics(self, paths, demos=None, prefix=''):
        """
        :param paths: list of dicts, one per gradient update. The keys of the dict are the env numb and the val the UNprocessed samples
        :param demos: same
        """
        logger.log("Saving visualization of paths")
        if not (type(paths) == list and len(paths) > 1 and type(paths[0]) == dict):
            raise ValueError("log_diagnostics only supports paths from vectorized sampler!")

        preupdate_paths = paths[0]  # paths obtained before any update
        postupdate_paths = paths[-1]  # paths obtained after all updates

        num_envs = len(preupdate_paths)

        for ind in range(min(5, num_envs)):
            plt.clf()
            plt.plot(*preupdate_paths[ind][0]['env_infos']['goal'][0], 'k*',
                     markersize=10)
            plt.hold(True)

            pre_points = preupdate_paths[ind][0]['observations']
            post_points = postupdate_paths[ind][0]['observations']
            plt.plot(pre_points[:, 0], pre_points[:, 1], '-r', linewidth=2)
            plt.plot(post_points[:, 0], post_points[:, 1], '-b', linewidth=1)

            # pre_points = preupdate_paths[ind][1]['observations']
            # post_points = postupdate_paths[ind][1]['observations']
            # plt.plot(pre_points[:, 0], pre_points[:, 1], '--r', linewidth=2)
            # plt.plot(post_points[:, 0], post_points[:, 1], '--b', linewidth=1)
            #

            if demos is not None:
                demo_paths = demos[0]
                demo_points = demo_paths[ind][0]['observations']
                plt.plot(demo_points[:, 0], demo_points[:, 1], '-.g', linewidth=2)

            plt.plot(0, 0, 'k.', markersize=5)
            plt.xlim([-0.8, 0.8])
            plt.ylim([-0.8, 0.8])
            plt.legend(['goal', 'pre-update path', 'post-update path', 'demos'])
            log_dir = logger.get_snapshot_dir()
            plt.savefig(osp.join(log_dir, 'post_update_plot' + str(ind) + '.png'))


class StraightDemo(Policy, Serializable):
    def __init__(self, *args, **kwargs):
        Serializable.quick_init(self, locals())
        super(StraightDemo, self).__init__(*args, **kwargs)

    @overrides
    def get_action(self, observation, objective_params, *args,
                   **kwargs):  # the same policy obj is applied to all envs!! so it needs to take in the goal!
        goal_vec = np.array(objective_params) - np.array(observation)
        return np.clip(goal_vec, *self.action_space.bounds), dict()

    @overrides
    def get_actions(self, observations, objective_params, *args, **kwargs):
        """
        needed for the vec env
        :param objective_params: this is an arg for the reset of the env! It specifies the task
        :param args/kwargs: these are just to throw away all other reset args for the env that don't define the task
        """
        actions = []
        for obs, goal in zip(observations, objective_params):
            goal_vec = np.array(goal) - np.array(obs)
            actions.append(np.clip(goal_vec, *self.action_space.bounds))
        return actions, dict()
