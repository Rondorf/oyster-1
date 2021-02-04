"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
import torch
from torch import nn as nn
from torch.nn import functional as F
from math import floor
from rlkit.policies.base import Policy
from rlkit.policies.exploration import *
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import PyTorchModule
from rlkit.torch.data_management.normalizer import TorchFixedNormalizer
from rlkit.torch.modules import LayerNorm
from rlkit.torch.distributions import TanhNormal
from torch.autograd import Variable


LOG_SIG_MAX = 2
LOG_SIG_MIN = -10



def identity(x):
    return x


class Mlp(PyTorchModule):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=ptu.identity,
            hidden_init=ptu.fanin_init,
            b_init_value=0.1,
            layer_norm=False,
            layer_norm_kwargs=None,
    ):
        self.save_init_params(locals())
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.fcs = nn.ModuleList([])
        self.layer_norms = nn.ModuleList([])
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, input, return_preactivations=False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output


class FlattenMlp(Mlp):
    """
    if there are multiple inputs, concatenate along dim 1
    """

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=1)
        return super().forward(flat_inputs, **kwargs)


class MlpPolicy(Mlp, Policy):
    """
    A simpler interface for creating policies.
    """

    def __init__(
            self,
            *args,
            obs_normalizer: TorchFixedNormalizer = None,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        self.obs_normalizer = obs_normalizer

    def forward(self, obs, **kwargs):
        if self.obs_normalizer:
            obs = self.obs_normalizer.normalize(obs)
        return super().forward(obs, **kwargs)

    def get_action(self, obs_np):
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

    def get_actions(self, obs):
        return self.eval_np(obs)


class TanhMlpPolicy(MlpPolicy):
    """
    A helper class since most policies have a tanh output activation.
    """
    def __init__(self, *args, **kwargs):
        self.save_init_params(locals())
        super().__init__(*args, output_activation=torch.tanh, **kwargs)


class MlpEncoder(FlattenMlp):
    '''
    encode context via MLP
    '''

    def reset(self, num_tasks=1):
        pass


class RecurrentEncoder(FlattenMlp):
    '''
    encode context via recurrent network
    '''

    def __init__(self,
                 *args,
                 **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        self.hidden_dim = self.hidden_sizes[-1]
        self.register_buffer('hidden', torch.zeros(1, 1, self.hidden_dim))

        # input should be (task, seq, feat) and hidden should be (task, 1, feat)

        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=1, batch_first=True)

    def forward(self, in_, return_preactivations=False):
        # expects inputs of dimension (task, seq, feat)
        task, seq, feat = in_.size()
        out = in_.view(task * seq, feat)

        # embed with MLP
        for i, fc in enumerate(self.fcs):
            out = fc(out)
            out = self.hidden_activation(out)

        out = out.view(task, seq, -1)
        out, (hn, cn) = self.lstm(out, (self.hidden, torch.zeros(self.hidden.size()).to(ptu.device)))
        self.hidden = hn
        # take the last hidden state to predict z
        out = out[:, -1, :]

        # output layer
        preactivation = self.last_fc(out)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output

    def reset(self, num_tasks=1):
        self.hidden = self.hidden.new_full((1, num_tasks, self.hidden_dim), 0)


class ConvNet(PyTorchModule):
    """
    if there are both image and vector inputs
    """
    def __init__(
            self,
            image_size,
            input_channels,
            conv_hidden_info,
            feature_dim,
            fc_hidden_sizes,
            additional_input_size,
            output_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=ptu.identity,
            hidden_init=ptu.fanin_init,
            b_init_value=0.1,
            layer_norm=False,
            layer_norm_kwargs=None,
            encoder=None
    ):
        self.save_init_params(locals())
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.image_size = image_size
        self.input_channels = input_channels
        self.output_size = output_size
        self.feature_dim = feature_dim
        self.additional_input_size = additional_input_size
        self.fc_hidden_sizes = fc_hidden_sizes
        self.convolution_info = conv_hidden_info
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm

        if encoder is None:
            self.conv_subnet = ImageEncoder(conv_hidden_info, feature_dim, image_size, input_channels)
        else:
            self.conv_subnet = encoder
        self.add_module('conv_subnet', self.conv_subnet)

        fc_in = self.feature_dim + self.additional_input_size
        self.fc_subnet = Mlp(fc_hidden_sizes, output_size, fc_in,
                             hidden_activation=hidden_activation,
                             output_activation=output_activation)
        self.add_module('fc_subnet', self.fc_subnet)

    def forward(self, *inputs, return_preactivations=False):
        image_input = inputs[0]
        additional_input = inputs[1]

        h1 = self.conv_subnet.forward(image_input)
        h1 = torch.flatten(h1, start_dim=1)
        h = torch.cat((h1, additional_input), 1)
        output, preactivation = self.fc_subnet.forward(h, return_preactivations=True)
        if return_preactivations:
            return output, preactivation
        else:
            return output


class ImageEncoder(PyTorchModule):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            image_size,
            input_channels,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=torch.tanh,
            hidden_init=ptu.fanin_init,
            b_init_value=0.1,
            batch_norm=False,
            layer_norm_kwargs=None,
    ):
        self.save_init_params(locals())
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_channels = input_channels
        self.input_size_x = image_size
        self.input_size_y = image_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.batch_norm = batch_norm
        self.convs = nn.ModuleList([])
        self.batch_norms = nn.ModuleList([])

        in_channels = self.input_channels
        x = self.input_size_x
        y = self.input_size_y

        # every hidden_sizes is in format [channels, kernel, stride]
        dilation = 1
        padding = 0

        for i, next_info in enumerate(hidden_sizes):
            next_channels = next_info[0]
            next_kernel = next_info[1]
            next_stride = next_info[2]
            x = floor(((x+2*padding-dilation*(next_kernel-1)-1)/next_stride)+1)
            y = floor(((y+2*padding-dilation*(next_kernel-1)-1)/next_stride)+1)

            conv = nn.Conv2d(in_channels, next_channels, next_kernel, stride=next_stride)
            in_channels = next_channels
            hidden_init(conv.weight)
            conv.bias.data.fill_(b_init_value)
            self.__setattr__("conv{}".format(i), conv)
            self.convs.append(conv)

            if self.batch_norm:
                ln = nn.BatchNorm2d(next_channels)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.batch_norms.append(ln)

        self.conv_output_size = x*y*next_channels

        self.last_fc = nn.Linear(self.conv_output_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)
        self.ln = nn.LayerNorm(self.output_size)

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(len(self.convs)):
            tie_weights(src=source.convs[i], trg=self.convs[i])

    def forward(self, input, return_preactivations=False):
        if input.max() > 1.:
            h = input / 255.
        else:
            h = input
        h_pre = input

        for i, conv in enumerate(self.convs):
            h_pre = conv(h)
            if self.batch_norm and i < len(self.convs) - 1:
                h_pre = self.batch_norms[i](h_pre)
            h = self.hidden_activation(h_pre)

        h = torch.flatten(h, start_dim=1)
        preactivation = self.last_fc(h)
        h_norm = self.ln(preactivation)
        output = self.output_activation(h_norm)

        if return_preactivations:
            return output, preactivation
        else:
            return output


class ConvEncoder(ConvNet):
    def reset(self, num_tasks=1):
        pass


class ConvTanhGaussianPolicy(ConvNet, ExplorationPolicy):
    """
    Usage:
    ```
    policy = TanhGaussianPolicy(...)
    action, mean, log_std, _ = policy(obs)
    action, mean, log_std, _ = policy(obs, deterministic=True)
    action, mean, log_std, log_prob = policy(obs, return_log_prob=True)
    ```
    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.
    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.
    """

    def __init__(
            self,
            image_size,
            input_channels,
            action_dim,
            fc_sizes,
            conv_sizes,
            feature_dim,
            std=None,
            additional_input_dim = 0,
            init_w=1e-3,
            encoder=None,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(
            image_size=image_size,
            input_channels=input_channels,
            conv_hidden_info=conv_sizes,
            feature_dim=feature_dim,
            fc_hidden_sizes=fc_sizes,
            additional_input_size=additional_input_dim,
            output_size=action_dim,
            init_w=init_w,
            encoder=encoder,
            **kwargs
        )
        self.obs_dim = (input_channels, image_size, image_size)
        self.action_dim = action_dim
        self.additional_input_dim = additional_input_dim

        self.log_std = None
        self.std = std
        if std is None:
            last_hidden_size = fc_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def get_action(self, obs, deterministic=False, additional_input=None):
        actions = self.get_actions(obs, deterministic=deterministic, additional_input=additional_input)
        return actions[0, :], {}

    @torch.no_grad()
    def get_actions(self, obs, deterministic=False, additional_input=None):
        outputs = self.forward(obs, deterministic=deterministic, additional_input=additional_input)[0]
        return np_ify(outputs)

    def forward(
            self,
            obs,
            reparameterize=True,
            deterministic=False,
            return_log_prob=False,
            additional_input=None
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        h = self.conv_subnet.forward(obs)
        h = torch.flatten(h, start_dim=1)

        if (additional_input is not None) and (self.additional_input_dim!=0):
            h = torch.cat((h, additional_input.float()), 1)

        for i, fc in enumerate(self.fc_subnet.fcs):
            h_pre = fc(h)
            if self.fc_subnet.layer_norm and i < len(self.fc_subnet.fcs) - 1:
                h_pre = self.fc_subnet.layer_norms[i](h_pre)
            h = self.fc_subnet.hidden_activation(h_pre)

        mean = self.fc_subnet.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            #log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            log_std = torch.tanh(log_std)
            log_std = LOG_SIG_MIN + 0.5 * (
                    LOG_SIG_MAX - LOG_SIG_MIN
            ) * (log_std + 1)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        log_prob = None
        if deterministic:
            action = torch.tanh(mean)
            if return_log_prob:
                tanh_normal = TanhNormal(mean, std)
                log_prob = tanh_normal.log_prob(action)
                log_prob = log_prob.sum(dim=1, keepdim=True)
        else:
            tanh_normal = TanhNormal(mean, std)
            if return_log_prob:
                if reparameterize:
                    action, pre_tanh_value = tanh_normal.rsample(
                        return_pretanh_value=True
                    )
                else:
                    action, pre_tanh_value = tanh_normal.sample(
                        return_pretanh_value=True
                    )
                log_prob = tanh_normal.log_prob(
                    action,
                    pre_tanh_value=pre_tanh_value
                )
                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                if reparameterize:
                    action = tanh_normal.rsample()
                else:
                    action = tanh_normal.sample()

        return action, mean, log_std, log_prob


def np_ify(tensor_or_other):
    if isinstance(tensor_or_other, Variable):
        return ptu.get_numpy(tensor_or_other)
    else:
        return tensor_or_other
