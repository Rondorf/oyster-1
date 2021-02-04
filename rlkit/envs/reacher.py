import gym
import dmc2gym
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from . import register_env


def rgb2gray(rgb):
    gray = 0.2989*rgb[0]+0.587*rgb[1]+0.114*rgb[2]
    return np.expand_dims(gray, axis=0)

@register_env('reacher')
class ReacherEnv(gym.Env):
    def __init__(self, image_size=64, action_repeat=4, max_episode_steps=200, n_tasks=2, dense=True, **kwargs):
        super(ReacherEnv, self).__init__()
        self.env = dmc2gym.make(
            domain_name='reacher',
            task_name='easy',
            seed=1,
            visualize_reward=False,
            from_pixels=True,
            height=image_size,
            width=image_size,
            frame_skip=action_repeat
        )

        self.dense = dense
        self.num_frames = 3
        shp = self.env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            #shape=((shp[0] * self.num_frames,) + shp[1:]),
            shape=((1 * self.num_frames,) + shp[1:]),
            dtype=self.env.observation_space.dtype
        )
        self._frames = deque([], maxlen=self.num_frames)
        self.env._max_episode_steps = max_episode_steps
        self.max_episode_steps = max_episode_steps
        self.action_space = self.env.action_space
        self.task_object = self.env.unwrapped._env.task
        self.goal_radius = 0.005
        self.task_object._target_size = self.goal_radius
        self.goals = [np.array([np.random.uniform(np.pi / 2, np.pi), 0.13]) for _ in range(n_tasks)]
        #np.random.uniform(.1, .13)
        #self.goals = [np.array([np.random.uniform(0, np.pi / 2), np.random.uniform(.1, .13)]) for _ in range(n_tasks)]
        #self.goals = [np.array([2.226, 0.122])]
        self.tasks = self.goals
        self.reset_task(0)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def get_task(self):
        """
        Return a task description, such as goal position or target velocity.
        """
        return self._goal

    def set_goal(self, goal):
        """
        Sets goal manually. Mainly used for reward relabelling.
        """
        self._goal = np.asarray(goal)
        self.task_object.angle = goal[0]
        self.task_object.radius = goal[1]
        range_min, range_max = self.env.physics.model.jnt_range[1]
        self.task_object.wrist_pos = np.random.uniform(range_min, range_max)
        self.task_object.shoulder_pos = np.random.uniform(-np.pi, np.pi)



        self.reset()

    def reset_task(self, idx=None):
        """
        Reset the task, either at random (if idx=None) or the given task.
        """
        if idx is None:
            self._goal = self.goals[int(np.random.randint(0,len(self.goals),1))]
            self.set_goal(self._goal)
        else:
            self._goal = self.goals[idx]
        self.set_goal(self._goal)
        self.reset()

    def step(self, action):
        """
        Execute one step in the environment.
        Should return: state, reward, done, info
        where info has to include a field 'task'.
        """
        obs, reward, done, info = self.env.step(action)
        self._frames.append(rgb2gray(obs))
        info['state'] = self.task_object.get_state(self.env.physics)
        if self.dense:
            info['reward_dense'] = self.reward(None, None)
            info['reward_sparse'] = self.task_object.get_sparse_reward(self.env.physics)
        return self._get_obs(), self.reward(None, None), done, info


    def reward(self, state, action):
        """
        Computes reward function of task.
        Returns the reward
        """
        if self.dense:
            return self.task_object.get_reward(self.env.physics)
        else:
            return self.task_object.get_sparse_reward(self.env.physics)

    def reset(self):
        """
        Reset the environment. This should *NOT* reset the task!
        Resetting the task is handled in the varibad wrapper (see wrappers.py).
        """
        obs = self.env.reset()
        for _ in range(self.num_frames):
            self._frames.append(rgb2gray(obs))
        return self._get_obs()

    def seed(self, seed=None):
        super(ReacherEnv, self).seed(seed)
        return self.env.seed(seed)

    def is_goal_state(self):
        if self.task_object.get_sparse_reward(self.env.physics) == 1:
            return True
        else:
            return False

    def _get_obs(self):
        assert len(self._frames) == self.num_frames
        return np.concatenate(list(self._frames), axis=0)

    def get_state(self):
        return self.task_object.get_state(self.env.physics)

    def get_all_task_idx(self):
        return range(len(self.goals))

    def plot_env(self):
        ax = plt.gca()
        # fix visualization
        plt.axis('scaled')
        # ax.set_xlim(-1.25, 1.25)
        ax.set_xlim(-0.32, 0.32)
        # ax.set_ylim(-0.25, 1.25)
        ax.set_ylim(-0.32, 0.32)
        ax.axhline(y=0, c='grey', ls='--')
        ax.axvline(x=0, c='grey', ls='--')
        plt.xticks([])
        plt.yticks([])
        goal_x = self._goal[1]*np.sin(self._goal[0])
        goal_y = self._goal[1]*np.cos(self._goal[0])
        circle = plt.Circle((goal_x, goal_y),
                            radius=self.goal_radius if hasattr(self, 'goal_radius') else 0.1,
                            alpha=0.3)
        ax.add_artist(circle)

    def plot_behavior(self, states, plot_env=True, **kwargs):
        self.plot_env()
        plt.plot(states[1:, 0], states[1:, 1], **kwargs)

    def reward_from_state(self, state, action):
        if self.dense:
            return self.task_object.get_reward_from_state_dense(self.env.physics, state)
        else:
            return self.task_object.get_reward_from_state_sparse(self.env.physics, state)

    def print_stuff(self):
        self.task_object.print_stuff(self.env.physics)
