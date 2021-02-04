"""
Launcher for experiments with PEARL

"""
import os
import pathlib
import numpy as np
import click
import json
import torch

from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.networks import *
from rlkit.torch.sac.sac import *
from rlkit.torch.sac.agent import PEARLAgent, PEARLImageAgent
from rlkit.launchers.launcher_util import setup_logger
import rlkit.torch.pytorch_util as ptu
from configs.default import default_config


def experiment(variant):
    # optional GPU mode
    #ptu.set_gpu_mode(variant['util_params']['use_gpu'], variant['util_params']['gpu_id'])

    # create multi-task environment and sample tasks
    env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))
    tasks = env.get_all_task_idx()
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    reward_dim = 1

    # instantiate networks
    latent_dim = variant['latent_size']
    context_encoder_input_dim = 2 * obs_dim + action_dim + reward_dim if variant['algo_params']['use_next_obs_in_context'] else obs_dim + action_dim + reward_dim
    context_encoder_output_dim = latent_dim * 2 if variant['algo_params']['use_information_bottleneck'] else latent_dim
    net_size = variant['net_size']
    recurrent = variant['algo_params']['recurrent']
    #encoder_model = RecurrentEncoder if recurrent else MlpEncoder

    context_encoder = ConvEncoder(
        image_size=70,
        input_channels=3,
        conv_hidden_info=[(32,3,2), (32,3,1), (32,3,1), (32,3,1)],
        feature_dim=50,
        fc_hidden_sizes=[1024, 1024],
        additional_input_size=action_dim + reward_dim,
        output_size=context_encoder_output_dim
    ).cuda()
    #context_encoder = encoder_model(
    #    hidden_sizes=[200, 200, 200],
    #    input_size=context_encoder_input_dim,
    #    output_size=context_encoder_output_dim,
    #)
    #qf1 = FlattenMlp(
    #    hidden_sizes=[net_size, net_size, net_size],
    #    input_size=obs_dim + action_dim + latent_dim,
    #    output_size=1,
    #)
    qf1 = ConvNet(
        image_size=70,
        input_channels=3,
        conv_hidden_info=[(32, 3, 2), (32, 3, 1), (32, 3, 1), (32, 3, 1)],
        feature_dim=50,
        fc_hidden_sizes=[1024, 1024],
        additional_input_size=action_dim + latent_dim,
        output_size=1
    ).cuda()
    #qf2 = FlattenMlp(
    #    hidden_sizes=[net_size, net_size, net_size],
    #    input_size=obs_dim + action_dim + latent_dim,
    #    output_size=1,
    #)
    mutual_encoder = qf1.conv_subnet
    qf2 = ConvNet(
        image_size=70,
        input_channels=3,
        conv_hidden_info=[(32, 3, 2), (32, 3, 1), (32, 3, 1), (32, 3, 1)],
        feature_dim=50,
        fc_hidden_sizes=[1024, 1024],
        additional_input_size=action_dim + latent_dim,
        output_size=1,
        encoder=mutual_encoder
    ).cuda()
    #vf = FlattenMlp(
    #    hidden_sizes=[net_size, net_size, net_size],
    #    input_size=obs_dim + latent_dim,
    #    output_size=1,
    #)
    vf = ConvNet(
        image_size=70,
        input_channels=3,
        conv_hidden_info=[(32, 3, 2), (32, 3, 1), (32, 3, 1), (32, 3, 1)],
        feature_dim=50,
        fc_hidden_sizes=[1024, 1024],
        additional_input_size=latent_dim,
        output_size=1,
        encoder=mutual_encoder
    ).cuda()
    #policy = TanhGaussianPolicy(
    #   hidden_sizes=[net_size, net_size, net_size],
    #    obs_dim=obs_dim + latent_dim,
    #    latent_dim=latent_dim,
    #    action_dim=action_dim,
    #)
    policy = ConvTanhGaussianPolicy(
        image_size=70,
        input_channels=3,
        action_dim=action_dim,
        fc_sizes=[1024, 1024],
        conv_sizes=[(32, 3, 2), (32, 3, 1), (32, 3, 1), (32, 3, 1)],
        feature_dim=50,
        additional_input_dim=latent_dim
    ).cuda()
    agent = PEARLImageAgent(
        latent_dim,
        context_encoder,
        policy,
        **variant['algo_params']
    ).cuda()
    algorithm = PEARLSoftActorCriticImage(
        env=env,
        train_tasks=list(tasks[:variant['n_train_tasks']]),
        eval_tasks=list(tasks[-variant['n_eval_tasks']:]),
        nets=[agent, qf1, qf2, vf],
        latent_dim=latent_dim,
        **variant['algo_params']
    )

    # optionally load pre-trained weights
    if variant['path_to_weights'] is not None:
        path = variant['path_to_weights']
        context_encoder.load_state_dict(torch.load(os.path.join(path, 'context_encoder.pth')))
        qf1.load_state_dict(torch.load(os.path.join(path, 'qf1.pth')))
        qf2.load_state_dict(torch.load(os.path.join(path, 'qf2.pth')))
        vf.load_state_dict(torch.load(os.path.join(path, 'vf.pth')))
        # TODO hacky, revisit after model refactor
        algorithm.networks[-2].load_state_dict(torch.load(os.path.join(path, 'target_vf.pth')))
        policy.load_state_dict(torch.load(os.path.join(path, 'policy.pth')))


    if ptu.gpu_enabled():
        algorithm.to()

    # debugging triggers a lot of printing and logs to a debug directory
    DEBUG = variant['util_params']['debug']
    os.environ['DEBUG'] = str(int(DEBUG))

    # create logging directory
    # TODO support Docker
    exp_id = 'debug' if DEBUG else None
    experiment_log_dir = setup_logger(variant['env_name'], variant=variant, exp_id=exp_id, base_log_dir=variant['util_params']['base_log_dir'])

    # optionally save eval trajectories as pkl files
    if variant['algo_params']['dump_eval_paths']:
        pickle_dir = experiment_log_dir + '/eval_trajectories'
        pathlib.Path(pickle_dir).mkdir(parents=True, exist_ok=True)

    # run the algorithm
    algorithm.train()

def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    # assume dicts have same keys
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to

@click.command()
@click.argument('config', default=None)
@click.option('--gpu', default=1)
@click.option('--docker', is_flag=True, default=False)
@click.option('--debug', is_flag=True, default=False)
def main(config, gpu, docker, debug):

    variant = default_config
    if config:
        with open(os.path.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    variant['util_params']['gpu_id'] = gpu

    experiment(variant)

if __name__ == "__main__":
    main()

