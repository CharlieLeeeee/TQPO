
import numpy as np
import torch

from rlpyt.models.mlp import MlpModel
from rlpyt.models.running_mean_std import RunningMeanStdModel
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.utils.collections import namedarraytuple

ValueInfo = namedarraytuple("ValueInfo", ["r_value", "c_value"])
RnnState = namedarraytuple("RnnState", ["h", "c"])


class tqpoModel(torch.nn.Module):

    def __init__(
            self,
            observation_shape,
            action_size,
            n_quantile,
            hidden_sizes=None,
            lstm_size=None,
            lstm_skip=True,
            constraint=True,
            hidden_nonlinearity="tanh",  # or "relu"
            mu_nonlinearity="tanh",
            init_log_std=0.,
            normalize_observation=True,
            var_clip=1e-6,
            ):
        super().__init__()
        self._n_quantile = n_quantile

        if hidden_nonlinearity == "tanh":  # So these can be strings in config file.
            hidden_nonlinearity = torch.nn.Tanh
        elif hidden_nonlinearity == "relu":
            hidden_nonlinearity = torch.nn.ReLU
        else:
            raise ValueError(f"Unrecognized hidden_nonlinearity string: {hidden_nonlinearity}")
        if mu_nonlinearity == "tanh":  # So these can be strings in config file.
            mu_nonlinearity = torch.nn.Tanh
        elif mu_nonlinearity == "relu":
            mu_nonlinearity = torch.nn.ReLU
        else:
            raise ValueError(f"Unrecognized mu_nonlinearity string: {mu_nonlinearity}")
        self._obs_ndim = len(observation_shape)
        input_size = int(np.prod(observation_shape))
        self.body = MlpModel(
            input_size=input_size,
            hidden_sizes=hidden_sizes or [256, 256],
            nonlinearity=hidden_nonlinearity,
        )
        last_size = self.body.output_size
        if lstm_size:
            lstm_input_size = last_size + action_size + 1
            self.lstm = torch.nn.LSTM(lstm_input_size, lstm_size)
            last_size = lstm_size
        else:
            self.lstm = None
        mu_linear = torch.nn.Linear(last_size, action_size)
        if mu_nonlinearity is not None:
            self.mu = torch.nn.Sequential(mu_linear, mu_nonlinearity())     # policy network
        else:
            self.mu = mu_linear
        self.value = torch.nn.Linear(last_size, 1)  # value network
        self.constraint = torch.nn.Linear(last_size, 1)    # c_value network

        self.log_std = torch.nn.Parameter(init_log_std *
            torch.ones(action_size))
        self._lstm_skip = lstm_skip
        if normalize_observation:
            print('rlpyt.projects.tqpo.tqpo_model.py | normalize_observation : ', True)
            self.obs_rms = RunningMeanStdModel(observation_shape)
            self.var_clip = var_clip
        self.normalize_observation = normalize_observation

    def forward(self, observation, prev_action, prev_reward, init_rnn_state=None):
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_ndim)
        if self.normalize_observation:
            obs_var = self.obs_rms.var
            if self.var_clip is not None:
                obs_var = torch.clamp(obs_var, min=self.var_clip)
            observation = torch.clamp((observation - self.obs_rms.mean) /
                obs_var.sqrt(), -10, 10)
        fc_x = self.body(observation.view(T * B, -1))
        if self.lstm is not None:
            lstm_inputs = [fc_x, prev_action, prev_reward]
            lstm_input = torch.cat([x.view(T, B, -1) for x in lstm_inputs],
                dim=2)
            init_rnn_state = None if init_rnn_state is None else tuple(init_rnn_state)
            lstm_out, (hn, cn) = self.lstm(lstm_input, init_rnn_state)
            lstm_out = lstm_out.view(T * B, -1)
            if self._lstm_skip:
                fc_x = fc_x + lstm_out
            else:
                fc_x = lstm_out

        mu = self.mu(fc_x)
        log_std = self.log_std.repeat(T * B, 1)
        r_value = self.value(fc_x).squeeze(-1)
        mu, log_std, r_value = restore_leading_dims((mu, log_std, r_value), lead_dim, T, B)

        c_value = self.constraint(fc_x).squeeze(-1)

        c_value = restore_leading_dims((c_value), lead_dim, T, B)
        value = ValueInfo(r_value=r_value, c_value=c_value)

        outputs = (mu, log_std, value)
        if self.lstm is not None:
            outputs += (RnnState(h=hn, c=cn),)

        return outputs

    def update_obs_rms(self, observation):
        if not self.normalize_observation:
            return
        self.obs_rms.update(observation)