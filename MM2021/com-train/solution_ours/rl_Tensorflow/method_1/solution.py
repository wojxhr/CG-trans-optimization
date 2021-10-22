"""
This demo aims to help player running system quickly by using the pypi library simple-emualtor https://pypi.org/project/simple-emulator/.
"""
from simple_emulator import CongestionControl

# We provided a simple algorithms about block selection to help you being familiar with this competition.
# In this example, it will select the block according to block's created time first and radio of rest life time to deadline secondly.
from simple_emulator import BlockSelection

from simple_emulator import create_emulator
import numpy as np

# for tf version < 2.0
import tensorflow as tf
import math

from simple_emulator import cal_qoe
import os, inspect
import random
# from train_net import Actor, Critic
import collections

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(1)
tf.set_random_seed(1)
EVENT_TYPE_FINISHED = 'F'
EVENT_TYPE_DROP = 'D'
EVENT_TYPE_TEMP = 'T'

# Superparameters
OUTPUT_GRAPH = False
MAX_EPISODE = 3000
DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 1000  # maximum time step in one episode
RENDER = False  # rendering wastes time
GAMMA = 0.9  # reward discount in TD error
LR_A = 0.0001  # learning rate for actor
LR_C = 0.001  # learning rate for critic
EPISODE = 10  # change every EPISODE times
N_F = 10  # speed, losepacket, application_speed
N_A = 5  # +100,0,-100
MAX_BANDWITH = 80000  # standardlize to 1
P = [1, 2/3, 1/3]
# NN_MODEL = None
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
NN_MODEL = current_dir+"/model/v1/nn_model_ep_3.ckpt"

CC_ADD_RATE = 100
CC_MINUS_RATE = 100


class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error

        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=128,  # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.acts_prob = tf.layers.dense(
                inputs=l1,
                units=n_actions,  # output units
                activation=tf.nn.softmax,  # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})  # get probabilities for all actions
        temp_p = probs.ravel()
        # print(temp_p)
        if np.isnan(temp_p[0]):
            temp_p[0] = random.uniform(0, 1.0)
            temp_p[1] = random.uniform(0, 1 - temp_p[0])
            temp_p[2] = random.uniform(0, 1 - temp_p[0] - temp_p[1])
            temp_p[3] = random.uniform(0, 1 - temp_p[0] - temp_p[1] - temp_p[2])
            temp_p[4] = 1 - temp_p[0] - temp_p[1] - temp_p[2] - temp_p[3]
        return np.random.choice(np.arange(probs.shape[1]), p=temp_p)  # return a int


class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=128,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.v = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + GAMMA * self.v_ - self.v
            self.loss = tf.square(self.td_error)  # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                    {self.s: s, self.v_: v_, self.r: r})
        return td_error


class RL(CongestionControl):

    def __init__(self):
        super(RL, self).__init__()
        self.USE_CWND = False
        self.send_rate = 10000
        self.cwnd = 1
        self.drop_tag = 0
        # RL
        # self.actor = actor_1
        # self.critic = critic_1
        self.last_state = np.zeros(N_F)
        self.last_action = 0
        self.reward = 0
        # state
        self.counter = 0
        self.cur_time = -1
        self.instant_drop_nums = 0
        self.instant_ack_nums = 0
        self.index = 0
        self.rtt = []
        self.eps_start_time = 0
        self.last_rtt = 0
        self.miss_deadline = 0
        self.last_inflight = collections.deque([0]*20)

    def estimate_bandwidth(self, cur_time, data):
        event_type = data["event_type"]
        event_time = cur_time

        if self.cur_time < event_time:
            self.drop_tag = 0

        if event_type == EVENT_TYPE_DROP:
            self.reward -= 0.5 * P[data["packet_information_dict"]['Block_info']['Priority']]
            self.instant_drop_nums += 1
            if self.drop_tag > 0:
                return
            self.counter += 1
            self.drop_tag += 1
            self.cur_time = event_time

        if event_type == EVENT_TYPE_FINISHED:
            self.reward += 0.5 * P[data["packet_information_dict"]['Block_info']['Priority']]
            self.instant_ack_nums += 1
            if event_time <= self.cur_time:
                return
            self.counter += 1
            self.cur_time = event_time

        # if self.cur_time > self.num:
        #     print(self.cur_time, data["packet_information_dict"]["Extra"])
        #     self.num += 0.5
        # if data["packet_information_dict"]["Extra"]["inflight"] > 400:
        #     self.send_rate = 750
        # elif data["packet_information_dict"]["Extra"]["inflight"] - sum(self.last_inflight) > 30:
        #     self.send_rate /= 1.1
        #     # print(self.cur_time)
        self.last_inflight.append(data["packet_information_dict"]["Extra"]["inflight"])
        self.last_inflight.popleft()

        self.rtt.append(data["packet_information_dict"]["Latency"] + data["packet_information_dict"]["Pacing_delay"])
        if event_time - data["packet_information_dict"]['Block_info']["Create_time"] > \
                data["packet_information_dict"]['Block_info']['Deadline']:
            self.miss_deadline += 1

        if self.counter == EPISODE:  # choose action every EPISODE times
            # print(self.send_rate)
            time_interval = cur_time - self.eps_start_time
            # print(data["event_time"], self.eps_start_time)
            # current_state
            state = np.zeros(N_F)
            # 发送速率
            state[0] = self.send_rate / MAX_BANDWITH
            # 接收速率
            state[1] = EPISODE / time_interval / MAX_BANDWITH
            # 平均RTT
            state[2] = np.mean(self.rtt) * 10
            # RTT的方差
            state[3] = np.var(self.rtt) * 100000
            state[4] = (np.mean(self.rtt) - self.last_rtt) * 100
            state[5] = self.instant_drop_nums / EPISODE
            state[6] = 1 - self.instant_drop_nums / EPISODE - 0.5
            state[7] = time_interval * 50
            state[8] = self.last_inflight[-1] / 200
            state[9] = (self.last_inflight[-1] - self.last_inflight[0]) / 50
            # print("%.4f"%state[5], "%.4f"%state[1], "%.4f"%state[2], "%.4f"%state[3],"%.4f"%state[4],"%.4f"%state[6])

            # choose action and explore
            a = actor.choose_action(state)
            if a == 0:
                self.send_rate *= 1.05
                if self.send_rate > MAX_BANDWITH * 2:
                    self.send_rate = MAX_BANDWITH * 2
            elif a == 1:
                self.send_rate += 100
                if self.send_rate > MAX_BANDWITH * 2:
                    self.send_rate = MAX_BANDWITH * 2
            elif a == 2:
                self.send_rate += 10
            elif a == 3:
                self.send_rate -= 100
                if self.send_rate < 400.0:
                    self.send_rate = 400.0
            else:
                self.send_rate /= 1.05
                if self.send_rate < 400.0:
                    self.send_rate = 400.0

            # episode_reward
            self.reward -= 0.9 * self.miss_deadline
            # print(self.reward)

            if self.index > 0:
                td_error = critic.learn(self.last_state, self.reward, state)
                actor.learn(self.last_state, self.last_action, td_error)
            self.last_state = state
            self.last_action = a
            self.reward = 0
            self.index += 1
            self.counter = 0
            self.last_rtt = np.mean(self.rtt)
            self.rtt = []
            self.eps_start_time = cur_time
            self.instant_drop_nums = 0
            self.miss_deadline = 0

    def append_input(self, data):
        self._input_list.append(data)

        if data["event_type"] != EVENT_TYPE_TEMP:
            self.cc_trigger(data)
            return {
                "cwnd": self.cwnd,
                "send_rate": self.send_rate
            }
        return None


# # we need a good teacher, so the teacher should learn faster than the actor
sess = tf.Session()
actor = Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A)
critic = Critic(sess, n_features=N_F, lr=LR_C)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=10000)
nn_model = NN_MODEL
if nn_model is not None:
    saver.restore(sess, nn_model)
#     print("Model restored.")


class MySolution(BlockSelection, RL):

    def select_block(self, cur_time, block_queue):
        '''
        The alogrithm to select the block which will be sended in next.
        The following example is selecting block by the create time firstly, and radio of rest life time to deadline secondly.
        :param cur_time: float
        :param block_queue: the list of Block.You can get more detail about Block in objects/block.py
        :return: int
        '''

        def is_better(block):
            best_block_create_time = best_block.block_info["Create_time"]
            cur_block_create_time = block.block_info["Create_time"]
            best_block_priority = best_block.block_info["Priority"]
            cur_block_priority = block.block_info["Priority"]
            best_block_size = best_block.block_info["Size"]
            block_size = block.block_info["Size"]
            best_block_rest = (best_block_size - best_block.offset * 1480)
            block_rest = (block_size - block.offset * 1480)
            best_block_ddl_rest = best_block.block_info["Deadline"] - (cur_time - best_block_create_time)
            block_ddl_rest = block.block_info["Deadline"] - (cur_time - cur_block_create_time)
            best_block_rest_ratio = best_block_rest / best_block_size
            block_rest_ratio = block_rest / block_size
            best_block_ddl_ratio = best_block_rest / best_block_ddl_rest
            block_ddl_radio = block_rest / block_ddl_rest
            best_block_rest_pkt = math.ceil(best_block_rest / 1480)
            block_rest_pkt = math.ceil(block_rest / 1480)

            # if block is miss ddl
            if (cur_time - cur_block_create_time) >= block.block_info["Deadline"]:
                return False
            # if best_block is miss ddl
            if (cur_time - best_block_create_time) >= best_block.block_info["Deadline"]:
                return True
            # block_rest is less
            if best_block_rest_ratio * best_block_rest_pkt * 2 <= block_rest_ratio * block_rest_pkt:
                return False
            # block_ddl_rest is less
            if best_block_ddl_rest / best_block_rest_pkt >= block_ddl_rest / block_rest_pkt * 1.5:
                return False
            # all information
            if best_block_ddl_ratio * best_block_rest_ratio * (0.01 + 0.5 * best_block_priority) > block_ddl_radio * \
                    block_rest_ratio * (0.01 + 0.5 * cur_block_priority):
                return True
            else:
                return False

        best_block_idx = -1
        best_block = None
        for idx, item in enumerate(block_queue):
            if best_block is None or is_better(item):
                best_block_idx = idx
                best_block = item

        return best_block_idx

    def on_packet_sent(self, cur_time):
        """
        The part of solution to update the states of the algorithm when sender need to send packet.
        """
        return super().on_packet_sent(cur_time)

    def cc_trigger(self, cur_time, event_info):
        """
        The part of algorithm to make congestion control, which will be call when sender get an event about acknowledge or lost from reciever.
        See more at https://github.com/AItransCompetition/simple_emulator/tree/master#congestion_control_algorithmpy.
        """
        # estimate the bandwidth
        super().estimate_bandwidth(cur_time, event_info)

        # set cwnd or sending rate in sender
        return {
            "cwnd": self.cwnd,
            "send_rate": self.send_rate,
        }


