
######################################################################
# Algorithm file.
######################################################################

import numpy as np
import torch
# import torch.nn.functional as F
from collections import namedtuple, deque

from rlpyt.algos.pg.base import PolicyGradientAlgo, OptInfo
from rlpyt.agents.base import AgentInputs
from rlpyt.utils.tensor import valid_mean
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.buffer import buffer_to, buffer_method
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.misc import iterate_mb_idxs
from rlpyt.algos.utils import (discount_return,
    generalized_advantage_estimation, valid_from_done)
from rlpyt.projects.tqpo.dist_rl_utils import (normalize, quantile_huber_loss, weibull_tail_loss, quantile_target_estimation,
                                                gae_quantile_simple, compute_prob_ratio)
from rlpyt.distributions.gaussian import DistInfoStd
from rlpyt.projects.tqpo.tqpo_model import RnnState
from rlpyt.utils.logging import logger


from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from scipy import integrate

LossInputs = namedarraytuple("LossInputs",
    ["agent_inputs", "action", "r_return", "r_advantage", "valid", "old_dist_info",
    "c_return"])
LossInfo = namedtuple("LossInfo", ("piRLoss", "piCLoss", "RvalueLoss", "CvalueLoss"))
OptInfoCost = namedtuple("OptInfoCost", OptInfo._fields + LossInfo._fields + ("costPenalty",
    "costLimit","q_quantile_lag","prob_outage","q_quantile_step", "tilt_pos","delta"))


class tqpo(PolicyGradientAlgo):
    """
    Quantile Constrained Policy Optimization

    """
    opt_info_fields = OptInfoCost._fields

    def __init__(
            self,
            discount=0.99,
            learning_rate=0.001,
            value_loss_coeff=1.,
            entropy_loss_coeff=0.,
            OptimCls=torch.optim.Adam,
            optim_kwargs=None,
            clip_grad_norm=1.,
            initial_optim_state_dict=None,
            gae_lambda=0.97,
            minibatches=1,
            epochs=8,
            ratio_clip=0.1,
            linear_lr_schedule=False,
            normalize_advantage=False,
            cost_discount=None,  # if None, defaults to discount.
            cost_gae_lambda=None,
            cost_value_loss_coeff=None,
            ep_cost_ema_alpha=0,  # 0 for hard update, 1 for no update.
            ep_outage_ema_alpha=0,  # 0 for hard update, 1 for no update.
            ep_cost_eqa_alpha=0,  # 0 for hard update, 1 for no update.
            objective_penalized=True,  # False for reward-only learning
            learn_c_value=True,  # Also False for reward-only learning
            penalty_init=1.,
            cost_limit=25,
            cost_scale=10.,  # divides; applied to raw cost and cost_limit
            target_outage_prob=0.3,
            weibull_tail_prob=0.3,
            n_quantile=25,
            normalize_cost_advantage=False,
            pid_Ki=0.1,
            sum_norm=True,  # L = (J_r - lam * J_c) / (1 + lam); lam <= 0
            diff_norm=False,  # L = (1 - lam) * J_r - lam * J_c; 0 <= lam <= 1
            penalty_max=100,
            step_cost_limit_steps=None,  # Change the cost limit partway through
            step_cost_limit_value=None,  # New value.
            reward_scale=1,  # multiplicative (unlike cost_scale)
            lagrange_quadratic_penalty=False,
            quadratic_penalty_coeff=1,
            new_T=128,
            new_B=104,
            q_lr=2e-1,
            lr_decay_freq=2e4,
            lr_decay_rate=0.9,
            ):
        assert learn_c_value or not objective_penalized
        assert (step_cost_limit_steps is None) == (step_cost_limit_value is None)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        cost_discount = discount if cost_discount is None else cost_discount
        cost_gae_lambda = (gae_lambda if cost_gae_lambda is None else
            cost_gae_lambda)
        cost_value_loss_coeff = (value_loss_coeff if cost_value_loss_coeff is
            None else cost_value_loss_coeff)
        if optim_kwargs is None:
            optim_kwargs = dict()
        save__init__args(locals())

        self.cost_limit /= self.cost_scale
        if step_cost_limit_value is not None:
            self.step_cost_limit_value /= self.cost_scale

        self.MSELoss = torch.nn.MSELoss()

        self.target_outage_prob = target_outage_prob
        self.target_outage_ind = np.floor(n_quantile * (1 - target_outage_prob) - 0.5).astype(int)

        self.weibull_tail_prob = weibull_tail_prob
        self.weibull_tail_ind = np.floor(n_quantile * (1 - weibull_tail_prob) - 0.5).astype(int)

        logger.log('self.target_outage_prob', self.target_outage_prob)
        logger.log('self.target_outage_ind', self.target_outage_ind)
        

        self.q_buffer_len = 100
        self.current_ep_costs = deque(maxlen=100)
        self.current_step_costs =deque(maxlen=1000*100)
        self.current_q_quantile_lag = deque(maxlen=self.q_buffer_len)

        self.q_quantile_step = 0.0
        self.delta=0.1

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)
        self._batch_size = self.batch_spec.size // self.minibatches  # For logging.
        if self.linear_lr_schedule:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=self.optimizer,
                lr_lambda=lambda itr: (self.n_itr - itr) / self.n_itr)  # Step once per itr.
            self._ratio_clip = self.ratio_clip  # Save base value.
        if self.step_cost_limit_steps is None:
            self.step_cost_limit_itr = None
        else:
            self.step_cost_limit_itr = int(self.step_cost_limit_steps //
                (self.batch_spec.size * self.world_size))
            # print("\n\n step cost itr: ", self.step_cost_limit_itr, "\n\n")
        self._ep_cost_ema = self.cost_limit  # No derivative at start.
        self._ep_outage_ema = self.target_outage_prob
        self._ep_cost_eqa = self.cost_limit
        self._ddp = self.agent._ddp
        assert self._ddp == False, print('ddp should be False')

        self.pid_i = self.cost_penalty = .5
  
    def transform_sample(self, sample):
        assert isinstance(sample, torch.Tensor), print(f'sample is not a tensor')
        old_shape = sample.shape
        assert old_shape[0] * old_shape[1] == self.new_T * self.new_B, print(f'old_shape[0] * old_shape[1] ({old_shape[0]} * {old_shape[1]}) does not match to new_T * new_B ({self.new_T} * {self.new_B})')
        if old_shape[0] == self.new_T and old_shape[1] == self.new_B:
            return sample

        new_shape = (self.new_B, self.new_T) + tuple(old_shape[2:]) if len(old_shape) > 2 else (self.new_B, self.new_T)
        return sample.transpose(0, 1).reshape(*new_shape).transpose(0, 1).contiguous()

    def indicator(self,x, y:torch.Tensor):
        return torch.where(y >= x, torch.ones_like(y), torch.zeros_like(y))

    def optimize_agent(self, itr, samples):
        print("tqpo_agent.py | optimize_agent")

        recurrent = self.agent.recurrent
        agent_inputs = AgentInputs(  # agent inputs, Move inputs to device once, index there.
            observation=self.transform_sample(samples.env.observation),
            prev_action=self.transform_sample(samples.agent.prev_action),
            prev_reward=self.transform_sample(samples.env.prev_reward),
        )
        action = self.transform_sample(samples.agent.action)    # action distribution
        dist_info = samples.agent.agent_info.dist_info
        old_dist_info = DistInfoStd(
            mean=self.transform_sample(dist_info.mean),
            log_std=self.transform_sample(dist_info.log_std)
        )
        
        agent_inputs = buffer_to(agent_inputs, device=self.agent.device)    # move action distribution to device

        opt_info = OptInfoCost(*([] for _ in range(len(OptInfoCost._fields))))

        (r_return, r_advantage,
         c_return, valid, ep_cost_quantile) = self.process_returns(itr, samples)  # process returns from samples

        # estimate quantile per epoch
        opt_info.q_quantile_lag.append(ep_cost_quantile.item())
        
        current_ep_cost = torch.tensor(list(self.current_ep_costs))     # estimate prob_outage by 100 episode cost
        ind=self.indicator(self.cost_limit*self.cost_scale, current_ep_cost)
        prob_outage=torch.mean(ind).detach()
        print("prob_outage: ",prob_outage.item())
        opt_info.prob_outage.append(prob_outage.item())
        
        
        step_cost = torch.tensor(list(self.current_step_costs))       # use this epoch's cost to update the q_quantile_step
        q_quantile_ind_step=np.floor(len(step_cost)*(1-self.target_outage_prob)).astype(int)-1
        q_quantile_step=np.sort(step_cost)[q_quantile_ind_step]
        opt_info.q_quantile_step.append(q_quantile_step.item())
        self.q_quantile_step=q_quantile_step.item()

        self.current_q_quantile_lag.append(ep_cost_quantile)        # use CDF to calculate the tilted rate
        list_q = list(self.current_q_quantile_lag)
        q_cdf = len(list(filter(lambda x: x < self.cost_limit, list_q)))/len(list_q) if len(list_q) > self.q_buffer_len else 0     
        self.tilt_pos = (q_cdf+self.delta)/(1+self.delta)
        self.tilt_neg = (1-q_cdf+self.delta)/(1+self.delta)

        opt_info.tilt_pos.append(self.tilt_pos)

        loss_inputs = LossInputs(
            agent_inputs=agent_inputs,
            action=action,
            r_return=r_return,
            r_advantage=r_advantage,
            valid=valid,
            old_dist_info=old_dist_info,
            c_return=c_return
        )

        if (self.step_cost_limit_itr is not None and
                self.step_cost_limit_itr == itr):
            self.cost_limit = self.step_cost_limit_value
        opt_info.costLimit.append(self.cost_limit)


        delta = float(ep_cost_quantile - self.cost_limit)   # update lambda
        delta = min(10., delta)
        if delta > 0:
            delta = self.tilt_pos * delta 
        else:
            delta = self.tilt_neg * delta
        delta = delta * self.pid_Ki
        self.pid_i = max(0., self.pid_i + delta)
        opt_info.delta.append(delta)

        self.cost_penalty = max(0., self.pid_i)

        opt_info.costPenalty.append(self.cost_penalty)


        if hasattr(self.agent, "update_obs_rms"):
            self.agent.update_obs_rms(agent_inputs.observation)
            if itr == 0:
                return opt_info

        if recurrent:
            prev_rnn_state = RnnState(h=self.transform_sample(samples.agent.agent_info.prev_rnn_state.h),
                                      c=self.transform_sample(samples.agent.agent_info.prev_rnn_state.c))
            init_rnn_state = prev_rnn_state[0]  # T=0.

        # If recurrent, use whole trajectories, only shuffle B; else shuffle all.
        batch_size = self.new_B if self.agent.recurrent else self.new_T * self.new_B
        mb_size = batch_size // self.minibatches

        for epo in range(self.epochs):
            for idxs in iterate_mb_idxs(batch_size, mb_size, shuffle=True):
                T_idxs = slice(None) if recurrent else idxs % self.new_T
                B_idxs = idxs if recurrent else idxs // self.new_T      # B is shuffled!
                self.optimizer.zero_grad()
                rnn_state = init_rnn_state[B_idxs] if recurrent else None
                # NOTE: if not recurrent, will lose leading T dim, should be OK.
                loss, entropy, pi_losses, value_losses = self.loss(*loss_inputs[T_idxs, B_idxs], rnn_state)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.agent.parameters(), self.clip_grad_norm)
                self.optimizer.step()

                opt_info.loss.append(loss.item())
                opt_info.gradNorm.append(grad_norm)
                opt_info.entropy.append(entropy.item())
                opt_info.piRLoss.append(pi_losses[0].numpy())
                opt_info.piCLoss.append(pi_losses[1].numpy())
                opt_info.RvalueLoss.append(value_losses[0].numpy())
                opt_info.CvalueLoss.append(value_losses[1].numpy())

                self.update_counter += 1

        if self.linear_lr_schedule:
            self.lr_scheduler.step()
            self.ratio_clip = self._ratio_clip * (self.n_itr - itr) / self.n_itr

        return opt_info

    def loss(self, agent_inputs, action, r_return, r_advantage, valid, old_dist_info,
            c_return, init_rnn_state=None):
        if init_rnn_state is not None:
            # [B,N,H] --> [N,B,H] (for cudnn).
            init_rnn_state = buffer_method(init_rnn_state, "transpose", 0, 1)
            init_rnn_state = buffer_method(init_rnn_state, "contiguous")
            dist_info, value, _rnn_state = self.agent(*agent_inputs, init_rnn_state)
        else:
            dist_info, value = self.agent(*agent_inputs)
        dist = self.agent.distribution

        ratio = dist.likelihood_ratio(action, old_dist_info=old_dist_info,
            new_dist_info=dist_info)
        
        self.ratio=ratio

        # cost loss
        ind=self.indicator(self.q_quantile_step, c_return.detach())
        adv = -ind/self.target_outage_prob*0.1  # normalize the cost loss
        surr_1 = ratio * adv
        clipped_ratio = torch.clamp(ratio, 1. - self.ratio_clip, 1. + self.ratio_clip)
        surr_2 = clipped_ratio * adv
        surrogate = torch.min(surr_1, surr_2)
        constraint_loss = - valid_mean(surrogate, valid)

        # reward loss
        surr_1 = ratio * r_advantage
        clipped_ratio = torch.clamp(ratio, 1. - self.ratio_clip,
            1. + self.ratio_clip)
        surr_2 = clipped_ratio * r_advantage
        surrogate = torch.min(surr_1, surr_2)
        pi_loss = - valid_mean(surrogate, valid)
        pi_r_loss = pi_loss.clone()

        # value net loss
        if self.reward_scale == 1.:
            value_error = value.r_value - r_return
        else:
            value_error = value.r_value - (r_return / self.reward_scale)
        value_se = 0.5 * value_error ** 2

        r_value_loss = self.value_loss_coeff * valid_mean(value_se, valid)

        entropy = dist.mean_entropy(dist_info, valid)
        entropy_loss = - self.entropy_loss_coeff * entropy


        if self.sum_norm:  # 1 / (1 + lam) * (R + lam * C) this is the default
            pi_loss += self.cost_penalty * constraint_loss
            pi_loss /= (1 + self.cost_penalty)
        else:
            pi_loss += self.cost_penalty * constraint_loss

        loss = pi_loss + entropy_loss + r_value_loss
        
        # cost value loss
        c_value_error = value.c_value - c_return
        c_value_se = 0.5 * c_value_error ** 2
        c_value_loss = valid_mean(c_value_se, valid)
        loss += self.cost_value_loss_coeff * c_value_loss

        pi_losses = (pi_r_loss.detach(), constraint_loss.detach())
        value_losses = (r_value_loss.detach(),c_value_loss.detach())

        return loss, entropy, pi_losses, value_losses

    def process_returns(self, itr, samples):
        reward, cost = self.transform_sample(samples.env.reward), self.transform_sample(samples.env.env_info.cost)
        
        cost /= self.cost_scale
        done = self.transform_sample(samples.env.done)
        r_value, c_value = samples.agent.agent_info.value  # A named 2-tuple.
        r_bv, c_bv = samples.agent.bootstrap_value  # A named 2-tuple.
        r_value = self.transform_sample(r_value)
        c_value = self.transform_sample(c_value)

        if not r_bv.shape[1] == self.new_B:
            r_bv_t, c_bv_t = [], []
            for i in range(r_bv.shape[1]):
                m = int(self.new_B // r_bv.shape[1])
                r_bv_t.extend([r_value[0, i * m + 1:(i + 1) * m].clone().unsqueeze(0), r_bv[:, i].clone().unsqueeze(0)])
                c_bv_t.extend([c_value[0, i * m + 1:(i + 1) * m].clone().unsqueeze(0), c_bv[:, i].clone().unsqueeze(0)])

            r_bv = torch.cat(r_bv_t, 1)
            c_bv = torch.cat(c_bv_t, 1)

        if self.reward_scale != 1:
            reward *= self.reward_scale
            r_value *= self.reward_scale
            r_bv *= self.reward_scale

        done = done.type(reward.dtype)
        
        _, c_return = generalized_advantage_estimation(cost, c_value, done, c_bv, self.cost_discount, self.cost_gae_lambda)

        if self.gae_lambda == 1:
            r_return = discount_return(reward, done, r_bv, self.discount)
            r_advantage = r_return - r_value
        else:
            r_advantage, r_return = generalized_advantage_estimation(reward, r_value, done, r_bv, self.discount, self.gae_lambda)

        self.current_step_costs.extend(c_return.detach().reshape(-1))

        if not self.mid_batch_reset or self.agent.recurrent:
            valid = valid_from_done(done)  # Recurrent: no reset during training.
            ep_cost_mask = torch.logical_and(valid, done)
        else:
            valid = None  # OR: torch.ones_like(done)
            ep_cost_mask = done

        cum_cost = self.transform_sample(samples.env.env_info.cum_cost)
        ep_costs = cum_cost[ep_cost_mask.type(torch.bool)]
        ep_outages = torch.gt(ep_costs / self.cost_scale, self.cost_limit).float()
        self.current_ep_costs.extend(ep_costs)

        if ep_costs.numel() > 0:  # Might not have any completed trajectories.
            ep_cost_avg = ep_costs.mean()
            ep_cost_avg /= self.cost_scale
            self._ep_cost_ema *= self.ep_cost_ema_alpha
            self._ep_cost_ema += (1 - self.ep_cost_ema_alpha) * ep_cost_avg


            # use 0.1 episode cost to update lambda
            quantile_ind = np.floor(len(self.current_ep_costs) * (1 - self.target_outage_prob)).astype(int)
            ep_cost_quantile = np.sort(self.current_ep_costs)[quantile_ind]     # choose the 0.1 quantile index
            ep_cost_quantile /= self.cost_scale
            self._ep_cost_eqa *= self.ep_cost_eqa_alpha
            self._ep_cost_eqa += (1 - self.ep_cost_eqa_alpha) * ep_cost_quantile    # EMA to update quantile

        if ep_outages.numel() > 0:  # Might not have any completed trajectories.
            ep_outage_avg = ep_outages.mean()
            self._ep_outage_ema *= self.ep_outage_ema_alpha
            self._ep_outage_ema += (1 - self.ep_outage_ema_alpha) * ep_outage_avg

        # Normalize advantages
        if self.normalize_advantage:
            r_advantage[:] = normalize(value=r_advantage, valid=valid)


        return (r_return, r_advantage, c_return, valid, self._ep_cost_eqa)

