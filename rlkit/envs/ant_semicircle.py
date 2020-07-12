import numpy as np

from rlkit.envs.ant_multitask_base import MultitaskAntEnv
from . import register_env


@register_env('ant-semicircle')
class AntSemiCircleEnv(MultitaskAntEnv):
    def __init__(self, task={}, n_tasks=2, randomize_tasks=True, **kwargs):
        super(AntSemiCircleEnv, self).__init__(task, n_tasks, **kwargs)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        xposafter = np.array(self.get_body_com("torso"))

        goal_reward = -np.sum(np.abs(xposafter[:2] - self._goal)) # make it happy, not suicidal

        ctrl_cost = .1 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 0.0
        reward = goal_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        done = False
        ob = self._get_obs()
        return ob, reward, done, dict(
            goal_forward=goal_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
        )

    def sample_tasks(self, num_tasks):
        # a = np.random.random(num_tasks) * 2 * np.pi
        a = np.array([np.random.uniform(0, np.pi) for _ in range(num_tasks)])
        # r = 3 * np.random.random(num_tasks) ** 0.5
        r = 1
        goals = np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)
        tasks = [{'goal': goal} for goal in goals]
        return tasks

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])


@register_env('sparse-ant-semicircle')
class SparseAntSemiCircleEnv(AntSemiCircleEnv):
    def __init__(self, task={}, n_tasks=2, randomize_tasks=True, goal_radius=0.2, **kwargs):
        self.goal_radius = goal_radius
        super().__init__(task, n_tasks, randomize_tasks, **kwargs)

    def sparsify_rewards(self, d):
        non_goal_reward_keys = []
        for key in d.keys():
            if key.startswith('reward') and key != "reward_goal":
                non_goal_reward_keys.append(key)
        non_goal_rewards = np.sum([d[reward_key] for reward_key in non_goal_reward_keys])
        sparse_goal_reward = 1. if self.is_goal_state() else 0.
        return non_goal_rewards + sparse_goal_reward

    def step(self, action):
        ob, reward, done, d = super().step(action)
        sparse_reward = self.sparsify_rewards(d)
        d.update({'sparse_reward': sparse_reward})
        return ob, reward, done, d

    def is_goal_state(self):
        xpos = np.array(self.get_body_com("torso"))
        if np.linalg.norm(xpos[:2] - self._goal) <= self.goal_radius:
            return True
        else:
            return False
