import os, inspect
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

from simple_emulator import PccEmulator, CongestionControl
from simple_emulator import Packet_selection
from simple_emulator import cal_qoe
# from config.constant import *
# from objects.cc_base import CongestionControl
MAX_CWND = 5000
MIN_CWND = 4

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
model_path = current_dir+"/model.pt"  # model3.pt

torch.manual_seed(1)
EVENT_TYPE_FINISHED='F'
EVENT_TYPE_DROP='D'
EVENT_TYPE_TEMP='T'
FixedCategorical = torch.distributions.Categorical

old_sample = FixedCategorical.sample
FixedCategorical.sample = lambda self: old_sample(self).unsqueeze(-1)

log_prob_cat = FixedCategorical.log_prob
FixedCategorical.log_probs = lambda self, actions: log_prob_cat(self, actions.squeeze(-1)).unsqueeze(-1)

FixedCategorical.mode = lambda self: self.probs.argmax(dim=1, keepdim=True)

FixedNormal = torch.distributions.Normal
log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: log_prob_normal(self, actions).sum(-1, keepdim=True)

entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: entropy(self).sum(-1)

FixedNormal.mode = lambda self: self.mean

# distributions
class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        init_ = lambda m: init(m,
              nn.init.orthogonal_,
              lambda x: nn.init.constant_(x, 0),
              gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(m,
              init_normc_,
              lambda x: nn.init.constant_(x, 0))

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())

# utils
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def init_normc_(weight, gain=1):
    weight.normal_(0, 1)
    weight *= gain / torch.sqrt(weight.pow(2).sum(1, keepdim=True))


# kfac
def _extract_patches(x, kernel_size, stride, padding):
    if padding[0] + padding[1] > 0:
        x = F.pad(x, (padding[1], padding[1], padding[0],
                      padding[0])).data  # Actually check dims
    x = x.unfold(2, kernel_size[0], stride[0])
    x = x.unfold(3, kernel_size[1], stride[1])
    x = x.transpose_(1, 2).transpose_(2, 3).contiguous()
    x = x.view(
        x.size(0), x.size(1), x.size(2),
        x.size(3) * x.size(4) * x.size(5))
    return x


def compute_cov_a(a, classname, layer_info, fast_cnn):
    batch_size = a.size(0)

    if classname == 'Conv2d':
        if fast_cnn:
            a = _extract_patches(a, *layer_info)
            a = a.view(a.size(0), -1, a.size(-1))
            a = a.mean(1)
        else:
            a = _extract_patches(a, *layer_info)
            a = a.view(-1, a.size(-1)).div_(a.size(1)).div_(a.size(2))
    elif classname == 'AddBias':
        is_cuda = a.is_cuda
        a = torch.ones(a.size(0), 1)
        if is_cuda:
            a = a.cuda()

    return a.t() @ (a / batch_size)


def compute_cov_g(g, classname, layer_info, fast_cnn):
    batch_size = g.size(0)

    if classname == 'Conv2d':
        if fast_cnn:
            g = g.view(g.size(0), g.size(1), -1)
            g = g.sum(-1)
        else:
            g = g.transpose(1, 2).transpose(2, 3).contiguous()
            g = g.view(-1, g.size(-1)).mul_(g.size(1)).mul_(g.size(2))
    elif classname == 'AddBias':
        g = g.view(g.size(0), g.size(1), -1)
        g = g.sum(-1)

    g_ = g * batch_size
    return g_.t() @ (g_ / g.size(0))


def update_running_stat(aa, m_aa, momentum):
    # Do the trick to keep aa unchanged and not create any additional tensors
    m_aa *= momentum / (1 - momentum)
    m_aa += aa
    m_aa *= (1 - momentum)


class SplitBias(nn.Module):
    def __init__(self, module):
        super(SplitBias, self).__init__()
        self.module = module
        self.add_bias = AddBias(module.bias.data)
        self.module.bias = None

    def forward(self, input):
        x = self.module(input)
        x = self.add_bias(x)
        return x


class KFACOptimizer(optim.Optimizer):
    def __init__(self,
                 model,
                 lr=0.25,
                 momentum=0.9,
                 stat_decay=0.99,
                 kl_clip=0.001,
                 damping=1e-2,
                 weight_decay=0,
                 fast_cnn=False,
                 Ts=1,
                 Tf=10):
        defaults = dict()

        def split_bias(module):
            for mname, child in module.named_children():
                if hasattr(child, 'bias') and child.bias is not None:
                    module._modules[mname] = SplitBias(child)
                else:
                    split_bias(child)

        split_bias(model)

        super(KFACOptimizer, self).__init__(model.parameters(), defaults)

        self.known_modules = {'Linear', 'Conv2d', 'AddBias'}

        self.modules = []
        self.grad_outputs = {}

        self.model = model
        self._prepare_model()

        self.steps = 0

        self.m_aa, self.m_gg = {}, {}
        self.Q_a, self.Q_g = {}, {}
        self.d_a, self.d_g = {}, {}

        self.momentum = momentum
        self.stat_decay = stat_decay

        self.lr = lr
        self.kl_clip = kl_clip
        self.damping = damping
        self.weight_decay = weight_decay

        self.fast_cnn = fast_cnn

        self.Ts = Ts
        self.Tf = Tf

        self.optim = optim.SGD(
            model.parameters(),
            lr=self.lr * (1 - self.momentum),
            momentum=self.momentum)

    def _save_input(self, module, input):
        if torch.is_grad_enabled() and self.steps % self.Ts == 0:
            classname = module.__class__.__name__
            layer_info = None
            if classname == 'Conv2d':
                layer_info = (module.kernel_size, module.stride,
                              module.padding)
            aa = compute_cov_a(input[0].data, classname, layer_info,
                               self.fast_cnn)

            # Initialize buffers
            if self.steps == 0:
                self.m_aa[module] = aa.clone()

            update_running_stat(aa, self.m_aa[module], self.stat_decay)

    def _save_grad_output(self, module, grad_input, grad_output):
        # Accumulate statistics for Fisher matrices
        if self.acc_stats:
            classname = module.__class__.__name__
            layer_info = None
            if classname == 'Conv2d':
                layer_info = (module.kernel_size, module.stride,
                              module.padding)

            gg = compute_cov_g(grad_output[0].data, classname, layer_info,
                               self.fast_cnn)

            # Initialize buffers
            if self.steps == 0:
                self.m_gg[module] = gg.clone()

            update_running_stat(gg, self.m_gg[module], self.stat_decay)

    def _prepare_model(self):
        for module in self.model.modules():
            classname = module.__class__.__name__
            if classname in self.known_modules:
                assert not ((classname in ['Linear', 'Conv2d']) and module.bias is not None), \
                                    "You must have a bias as a separate layer"

                self.modules.append(module)
                module.register_forward_pre_hook(self._save_input)
                module.register_backward_hook(self._save_grad_output)

    def step(self):
        # Add weight decay
        if self.weight_decay > 0:
            for p in self.model.parameters():
                p.grad.data.add_(self.weight_decay, p.data)

        updates = {}
        for i, m in enumerate(self.modules):
            assert len(list(m.parameters())
                       ) == 1, "Can handle only one parameter at the moment"
            classname = m.__class__.__name__
            p = next(m.parameters())

            la = self.damping + self.weight_decay

            if self.steps % self.Tf == 0:
                # My asynchronous implementation exists, I will add it later.
                # Experimenting with different ways to this in PyTorch.
                self.d_a[m], self.Q_a[m] = torch.symeig(
                    self.m_aa[m], eigenvectors=True)
                self.d_g[m], self.Q_g[m] = torch.symeig(
                    self.m_gg[m], eigenvectors=True)

                self.d_a[m].mul_((self.d_a[m] > 1e-6).float())
                self.d_g[m].mul_((self.d_g[m] > 1e-6).float())

            if classname == 'Conv2d':
                p_grad_mat = p.grad.data.view(p.grad.data.size(0), -1)
            else:
                p_grad_mat = p.grad.data

            v1 = self.Q_g[m].t() @ p_grad_mat @ self.Q_a[m]
            v2 = v1 / (
                self.d_g[m].unsqueeze(1) * self.d_a[m].unsqueeze(0) + la)
            v = self.Q_g[m] @ v2 @ self.Q_a[m].t()

            v = v.view(p.grad.data.size())
            updates[p] = v

        vg_sum = 0
        for p in self.model.parameters():
            v = updates[p]
            vg_sum += (v * p.grad.data * self.lr * self.lr).sum()

        nu = min(1, math.sqrt(self.kl_clip / vg_sum))

        for p in self.model.parameters():
            v = updates[p]
            p.grad.data.copy_(v)
            p.grad.data.mul_(nu)

        self.optim.step()
        self.steps += 1


# a2c_acktr
class A2C_ACKTR():
    def __init__(self,
                 actor_critic,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 alpha=None,
                 max_grad_norm=None,
                 acktr=False):

        self.actor_critic = actor_critic
        self.acktr = acktr

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm

        self.old_action_log_probs = torch.Tensor([8000, 1])

        if acktr:
            self.optimizer = KFACOptimizer(actor_critic)
        else:
            self.optimizer = optim.RMSprop(
                actor_critic.parameters(), lr, eps=eps, alpha=alpha)

    def update(self, rollouts):
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()
        old_values = rollouts.value_preds[:-1].view(num_steps, num_processes, 1)

        values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
            rollouts.obs[:-1].view(-1, *obs_shape),
            rollouts.recurrent_hidden_states[0].view(-1, self.actor_critic.recurrent_hidden_state_size),
            rollouts.masks[:-1].view(-1, 1),
            rollouts.actions.view(-1, action_shape))
        values = values.view(num_steps, num_processes, 1)
        vpredclipped = old_values + np.clip((values - old_values).detach(), - 0.2, 0.2)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)
        self.old_action_log_probs = action_log_probs

        # modified loss:
        # advantages1 = (rollouts.returns[:-1] - values).detach().pow(2)
        # advantages2 = (rollouts.returns[:-1] - vpredclipped).detach().pow(2)
        # advantages = np.maximum(advantages1, advantages2)
        # value_loss = advantages.mean()
        #
        # # advantages = rollouts.returns[:-1] - values
        # # value_loss = advantages.pow(2).mean()
        #
        # ratio = torch.Tensor.exp(self.old_action_log_probs - action_log_probs)
        # # action_loss1 = -(advantages.detach() * action_log_probs * ratio)
        # ratio = np.clip(ratio.detach(), 0.8, 1.2)
        # # action_loss2 = -(advantages.detach() * action_log_probs * ratio)
        # # action_loss = np.maximum(action_loss1.detach(), action_loss2.detach()).mean()
        # ppd = (self.old_action_log_probs - action_log_probs).pow(2).mean()
        # # action_loss = action_loss1.detach().mean()
        #
        # action_loss = -(advantages.detach() * action_log_probs * ratio).mean()

        # original loss:
        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = rollouts.returns[:-1] - values
        value_loss = advantages.pow(2).mean()

        action_loss = -(advantages.detach() * action_log_probs).mean()

        if self.acktr and self.optimizer.steps % self.optimizer.Ts == 0:
            # Sampled fisher, see Martens 2014
            self.actor_critic.zero_grad()
            pg_fisher_loss = -action_log_probs.mean()

            value_noise = torch.randn(values.size())
            if values.is_cuda:
                value_noise = value_noise.cuda()

            sample_values = values + value_noise
            vf_fisher_loss = -(values - sample_values.detach()).pow(2).mean()

            fisher_loss = pg_fisher_loss + vf_fisher_loss
            self.optimizer.acc_stats = True
            fisher_loss.backward(retain_graph=True)
            self.optimizer.acc_stats = False

        self.optimizer.zero_grad()
        (value_loss * self.value_loss_coef + action_loss -
         dist_entropy * self.entropy_coef).backward()
        # (value_loss * self.value_loss_coef + action_loss -
        #  dist_entropy * self.entropy_coef + 0.5 * ppd).backward()

        if self.acktr == False:
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                     self.max_grad_norm)

        self.optimizer.step()

        return value_loss.item(), action_loss.item(), dist_entropy.item()


# model
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}

        if len(obs_shape) == 1:
            self.base = MLPBase(obs_shape[0], **base_kwargs)
        else:
            raise NotImplementedError

        num_outputs = action_space
        self.dist = Categorical(self.base.output_size, num_outputs)

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class NNBase(nn.Module):

    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRUCell(recurrent_input_size, hidden_size)
            nn.init.orthogonal_(self.gru.weight_ih.data)
            nn.init.orthogonal_(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x = hxs = self.gru(x, hxs * masks)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N, 1)

            outputs = []
            for i in range(T):
                hx = hxs = self.gru(x[i], hxs * masks[i])
                outputs.append(hx)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.stack(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)

        return x, hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m,
            init_normc_,
            lambda x: nn.init.constant_(x, 0))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh()
        )

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh()
        )

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs


class RL(CongestionControl):

    def __init__(self):
        super(RL, self).__init__()
        self.USE_CWND = True
        # self.send_rate = 500.0
        self.send_rate = float("inf")
        self.cwnd = 1

        self.ssthresh = float("inf")
        self.curr_state = "slow_start"
        self.states = ["slow_start", "congestion_avoidance", "fast_recovery"]
        # the number of lost packets
        self.drop_nums = 0
        # the number of acknowledgement packets
        self.ack_nums = 0

        # current time
        self.cur_time = -1
        # the value of cwnd at last packet event
        self.last_cwnd = 0
        # the number of lost packets received at the current moment
        self.instant_drop_nums = 0
        self.miss_block_list = []

        torch.set_num_threads(1)
        device = torch.device("cpu")
        self.actor_critic = Policy((6,), 3, base_kwargs={'recurrent': None})
        self.actor_critic.to(device)
        self.agent = A2C_ACKTR(self.actor_critic, value_loss_coef=0.1, entropy_coef=0.01, acktr=True)
        self.actor_critic.load_state_dict(torch.load(model_path))
        self.flag = True

    def cc_trigger(self, data):
        event_type = data["event_type"]
        event_time = data["event_time"]
        if event_time - data["packet_information_dict"]['Block_info']['Create_time'] > \
                data["packet_information_dict"]['Block_info']['Deadline']:
            if data["packet_information_dict"]['Block_info']["Block_id"] not in self.miss_block_list:
                self.miss_block_list.append(data["packet_information_dict"]['Block_info']["Block_id"])

        latency = data["packet_information_dict"]["Latency"]
        pacing_delay = data["packet_information_dict"]["Pacing_delay"]
        rtt = latency + pacing_delay

        if self.cur_time < event_time:
            # initial parameters at a new moment
            self.last_cwnd = 0
            self.instant_drop_nums = 0

        # if self.flag and (rtt > 0.04 or self.cwnd > self.ssthresh):
        #     self.curr_state = self.states[1]
        #     self.cwnd = max(self.cwnd // 2, 1)
        #     self.flag = False
        #     self.ack_nums = 0
            # print(self.cwnd)

        # if packet is dropped
        if event_type == EVENT_TYPE_DROP:
            # dropping more than one packet at a same time is considered one event of packet loss
            if self.instant_drop_nums > 0:
                return
            self.instant_drop_nums += 1
            # step into fast recovery state
            # self.curr_state = self.states[2]
            self.drop_nums += 1
            # clear acknowledgement count
            self.ack_nums = 0
            # Ref 1 : For ensuring the event type, drop or ack?
            self.cur_time = event_time
            # if self.last_cwnd > 0 and self.last_cwnd != self.cwnd:
            #     # rollback to the old value of cwnd caused by acknowledgment first
            #     self.cwnd = self.last_cwnd
            #     self.last_cwnd = 0
            # if self.flag:
            #     self.curr_state = self.states[1]
            #     self.cwnd = math.ceil(self.cwnd / 2)
            #     self.flag = False
            #     print("cwnd,", self.cwnd)
            if self.drop_nums == 1:
                self.ssthresh = max(self.cwnd // 1.01, 1)
                self.cwnd = self.ssthresh
                self.curr_state = self.states[1]

        # if packet is acknowledged
        elif event_type == EVENT_TYPE_FINISHED:
            # Ref 1
            if event_time <= self.cur_time:
                return
            self.cur_time = event_time
            self.last_cwnd = self.cwnd
            # increase the number of acknowledgement packets
            self.ack_nums += 1
            # double cwnd in slow_start state
            if self.curr_state == self.states[0]:
                if self.ack_nums == self.cwnd:
                    self.cwnd *= 9
                    self.ack_nums = 0
                    self.cwnd = math.ceil(self.cwnd)
                    # (self.cwnd, rtt)
                # step into congestion_avoidance state due to exceeding threshhold
                if self.cwnd >= self.ssthresh:
                    self.curr_state = self.states[1]

            # increase cwnd linearly in congestion_avoidance state
            elif self.curr_state == self.states[1]:
                if self.ack_nums == self.cwnd:
                    self.ack_nums = 0
                    # loss rate
                    sum_loss_rate = sum([1 for data in self._input_list if data["event_type"] == 'D']) / len(
                        self._input_list)
                    instant_packet = list(
                        filter(lambda item: self._input_list[-1]["event_time"] - item["event_time"] < 0.1,
                               self._input_list))
                    instant_loss_rate = sum([1 for data in instant_packet if data["event_type"] == 'D']) / len(
                        instant_packet) if len(instant_packet) > 0 else 0
                    # throughput
                    sum_rate = sum([1 for data in self._input_list if data["event_type"] == 'F']) / (
                            self._input_list[-1]["event_time"] - self._input_list[0]["event_time"]) if len(
                        self._input_list) > 1 else 0
                    instant_rate = sum([1 for data in instant_packet if data["event_type"] == 'F']) / (
                            instant_packet[-1]["event_time"] - instant_packet[0]["event_time"]) if len(
                        instant_packet) > 1 else 0
                    instant_success_rate = sum([1 for data in instant_packet if data["event_type"] == 'F']) / len(
                        instant_packet) if len(instant_packet) > 1 else 0

                    s_ = []
                    s_.append(self.cwnd)
                    s_.append(rtt)
                    s_.append(sum_loss_rate)
                    s_.append(instant_loss_rate)
                    s_.append(sum_rate)
                    s_.append(instant_rate)
                    obs = torch.Tensor([s_])
                    recurrent_hidden_states = torch.FloatTensor([[0.]])
                    masks = torch.FloatTensor([[0.0]])

                    with torch.no_grad():
                        value, action, action_log_prob, recurrent_hidden_states = self.actor_critic.act(
                            obs,
                            recurrent_hidden_states,
                            masks)
                        a = action.item()  # 将Tensor类型的数据转化为Int型

                    # if random.random() < 0.1:
                    #     a = random.randint(0, 2)
                    if self.cwnd > MAX_CWND:
                        self.cwnd = MAX_CWND
                    elif self.cwnd < MIN_CWND:
                        self.cwnd = MIN_CWND
                    elif a == 0:
                        self.cwnd += 7
                    elif a == 1:
                        self.cwnd *= 1
                    else:
                        self.cwnd /= 1.01
                    self.cwnd = int(self.cwnd)
                    # print(event_time, self.cwnd, rtt, round(instant_rate, 2), instant_loss_rate)

        # reset threshhold and cwnd in fast_recovery state
        # if self.curr_state == self.states[2]:
        #     self.ssthresh = max(self.cwnd // 2, 1)
        #     self.cwnd = self.ssthresh
        #     self.curr_state = self.states[1]

    def append_input(self, data):
        # if data["packet_information_dict"]['Block_info']["Block_id"] not in self.miss_block_list:
        self._input_list.append(data)

        if data["event_type"] != EVENT_TYPE_TEMP:
            self.cc_trigger(data)
            return {
                "cwnd" : self.cwnd,
                "send_rate" : self.send_rate
            }
        return None

class MySolution(Packet_selection, RL):

    def select_packet(self, cur_time, packet_queue):
        """
        The algorithm to select which packet in 'packet_queue' should be sent at time 'cur_time'.
        The following example is selecting packet by the create time firstly, and radio of rest life time to deadline secondly.
        See more at https://github.com/AItransCompetition/simple_emulator/tree/master#packet_selectionpy.
        :param cur_time: float
        :param packet_queue: the list of Packet.You can get more detail about Block in objects/packet.py
        :return: int
        """
        def is_better(packet):
            best_block_create_time = best_packet.block_info["Create_time"]
            packet_block_create_time = packet.block_info["Create_time"]
            best_block_priority = best_packet.block_info["Priority"]
            packet_block_priority = packet.block_info["Priority"]
            best_block_size = best_packet.block_info["Size"]
            block_size = packet.block_info["Size"]
            best_block_rest = (best_block_size - best_packet.offset * 1480)
            block_rest = (block_size - packet.offset * 1480)
            best_block_ddl_rest = best_packet.block_info["Deadline"] - (cur_time - best_block_create_time)
            block_ddl_rest = packet.block_info["Deadline"] - (cur_time - packet_block_create_time)
            best_block_rest_ratio = best_block_rest / best_block_size
            block_rest_ratio = block_rest / block_size
            best_block_ddl_ratio = best_block_rest / best_block_ddl_rest
            block_ddl_radio = block_rest / block_ddl_rest
            best_block_rest_pkt = math.ceil(best_block_rest / 1480)
            block_rest_pkt = math.ceil(block_rest / 1480)

            # if packet is miss ddl
            if (cur_time - packet_block_create_time) >= packet.block_info["Deadline"]:
                return False
            # if best_block is miss ddl
            if (cur_time - best_block_create_time) >= best_packet.block_info["Deadline"]:
                return True
            # block_rest is less
            if best_block_rest_ratio * best_block_rest_pkt * 2 <= block_rest_ratio * block_rest_pkt:
                return False
            # # block_ddl_rest is less
            if best_block_ddl_rest / best_block_rest_pkt >= block_ddl_rest / block_rest_pkt * 1.5:
                return False
            # all information
            if best_block_ddl_ratio * best_block_rest_ratio * (1 + 0.4 * best_block_priority) > block_ddl_radio * \
                    block_rest_ratio * (1 + 0.4 * packet_block_priority):
                return True
            else:
                return False
        best_packet_idx = -1
        best_packet = None
        for idx, item in enumerate(packet_queue):
            if best_packet is None or is_better(item) :
                best_packet_idx = idx
                best_packet = item

        return best_packet_idx

    def make_decision(self, cur_time):
        """
        The part of algorithm to make congestion control, which will be call when sender need to send pacekt.
        See more at https://github.com/AItransCompetition/simple_emulator/tree/master#congestion_control_algorithmpy.
        """
        return super().make_decision(cur_time)

    def append_input(self, data):
        """
        The part of algorithm to make congestion control, which will be call when sender get an event about acknowledge or lost from reciever.
        See more at https://github.com/AItransCompetition/simple_emulator/tree/master#congestion_control_algorithmpy.
        """
        return super().append_input(data)


def path_cwd(traces_dir, blocks_dir):
    traces_list = os.listdir(traces_dir)
    for i in range(len(traces_list)):
        traces_list[i] = traces_dir + '/' + traces_list[i]
    blocks_list = os.listdir(blocks_dir)
    for i in range(len(blocks_list)):
        blocks_list[i] = blocks_dir + '/' + blocks_list[i]
    return traces_list, blocks_list


# if __name__ == '__main__':
#     # trace_list = ["traces/traces_2.txt", "traces/traces_3.txt", "traces/traces_22.txt", "traces/traces_23.txt",
#     #               "traces/traces_42.txt", "traces/traces_43.txt", "traces/traces_62.txt", "traces/traces_63.txt",
#     #               "traces/traces_82.txt", "traces/traces_83.txt", "traces/traces_102.txt", "traces/traces_103.txt"]
#     trace_dir_list = ["train/day_1/networks", "train/day_2/networks", "train/day_3/networks", "train/day_4/networks",
#                       "train/day_4/networks", "train/day_4/networks"]
#     block_dir_list = ["train/day_1/blocks", "train/day_2/blocks", "train/day_3/blocks", "train/day_4/blocks/blocks_1",
#                       "train/day_4/blocks/blocks_2", "train/day_4/blocks/blocks_3"]
#     score_list = []
#     for i in range(len(trace_dir_list)):
#         score = 0
#         trace_list, block_list = path_cwd(trace_dir_list[i], block_dir_list[i])
#         for trace_file in trace_list:
#             # The file path of packets' log
#             log_packet_file = "output/packet_log/packet-0.log"
#
#             # Use the object you created above
#             my_solution = MySolution()
#             # Create the emulator using your solution
#             # Specify USE_CWND to decide whether or not use crowded windows. USE_CWND=True by default.
#             # Specify ENABLE_LOG to decide whether or not output the log of packets. ENABLE_LOG=True by default.
#             # You can get more information about parameters at https://github.com/AItransCompetition/simple_emulator/tree/master#constant
#             emulator = PccEmulator(
#                 block_file=block_list,
#                 trace_file=trace_file,
#                 solution=my_solution,
#                 SEED=1,
#                 ENABLE_LOG=False
#             )
#
#             # Run the emulator and you can specify the time for the emualtor's running.
#             # It will run until there is no packet can sent by default.
#             emulator.run_for_dur(20)
#
#             # print the debug information of links and senders
#             # emulator.print_debug()
#
#             # torch.save(my_solution.actor_critic, "./models/model2.pt")
#             # Output the picture of emulator-analysis.png
#             # You can get more information from https://github.com/AItransCompetition/simple_emulator/tree/master#emulator-analysispng.
#             # analyze_pcc_emulator(log_packet_file, file_range="all")
#
#             # plot_rate(log_packet_file, trace_file="traces/trace.txt", file_range="all")
#             print(trace_file)
#             score += cal_qoe()
#             print(cal_qoe())
#             print()
#         print("sum:", score)
#         score_list.append(score)
#     print(score_list, sum(score_list))

if __name__ == '__main__':
    traces_list, blocks_list = path_cwd("train/day_3/networks", "train/day_3/blocks")
    qoe_sum = 0

    for trace_file in traces_list:
        # The file path of packets' log
        log_packet_file = "output/packet_log/packet-0.log"
        # Use the object you created above
        my_solution = MySolution()
        # Create the emulator using your solution
        # Specify USE_CWND to decide whether or not use crowded windows. USE_CWND=True by default.
        # Specify ENABLE_LOG to decide whether or not output the log of packets. ENABLE_LOG=True by default.
        emulator = PccEmulator(
            block_file=blocks_list,
            trace_file=trace_file,
            solution=my_solution,
            SEED=1,
            ENABLE_LOG=False
        )
        # Run the emulator and you can specify the time for the emualtor's running.
        # It will run until there is no packet can sent by default.
        emulator.run_for_dur(20)
        # print the debug information of links and senders
        # emulator.print_debug()
        # torch.save(my_solution.actor_critic, "./models/model2.pt")
        # analyze_pcc_emulator(log_packet_file, file_range="all")
        # plot_rate(log_packet_file, trace_file="traces/trace.txt", file_range="all")
        print(cal_qoe())
        qoe_sum += cal_qoe()

    print(qoe_sum/2)





