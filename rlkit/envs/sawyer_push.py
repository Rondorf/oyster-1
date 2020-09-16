import random
import numpy as np
import metaworld
from metaworld.envs.mujoco.sawyer_xyz.v1.sawyer_reach_push_pick_place import SawyerReachPushPickPlaceEnv
from . import register_env


@register_env('sawyer-push')
class SawyerPushEnv(SawyerReachPushPickPlaceEnv):
    def __init__(self, task={}, n_tasks=2, max_episode_steps=150, **kwargs):
        self.ml1 = metaworld.ML1('push-v1')
        self.env = self.ml1.train_classes['push-v1']()

        super(SawyerPushEnv, self).__init__()
        self._max_episode_steps = max_episode_steps
        self.task_type = 'push'

        self._goals = self.get_all_push_tasks()   # goals here are 6D (first 3 - obj_pos, final 3 - goal_pos)

        self._task = task
        self.tasks = self.sample_tasks(n_tasks)
        self.reset_task(0)

    # def step(self, action):
    #     raise NotImplementedError

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        self.set_task(idx)
        self._task = self.tasks[idx]
        self._goal = self._task['goal_pos']
        self.reset()

    # def reset_model(self):
    #     raise NotImplementedError

    def reward(self, state, action):
        raise NotImplementedError

    def set_task(self, idx):
        ''' task is idx from self.get_all_task_idx() '''
        super().set_task(self.tasks[idx]['task_id'])

    def sample_tasks(self, n_tasks):
        np.random.seed(1337)
        tasks_idx = np.random.permutation(range(len(self._goals)))[:n_tasks]
        tasks = [self._goals[str(idx)] for idx in tasks_idx]
        return tasks

    # def _get_obs(self):
    #     raise NotImplementedError

    # def get_task(self):
    #     return super()._get_pos_goal()

    # @staticmethod
    def get_all_push_tasks(self):
        # ml1 = metaworld.ML1('push-v1')
        # env = ml1.train_classes['push-v1']()
        goals = {}
        for i, task in enumerate(self.ml1.train_tasks):
            self.env.set_task(task)
            _ = self.env.reset()
            obj_pos = self.env._get_pos_objects()
            goal_pos = self.env._get_pos_goal()

            # goal = np.hstack((obj_pos, goal_pos))
            # goals.append(goal)
            goals[str(i)] = {'obj_pos': obj_pos, 'goal_pos': goal_pos, 'task_id': task}

        return goals
