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
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import math

from simple_emulator import cal_qoe
import os
import random
from train_net import Actor, Critic
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
EPISODE = 25  # change every EPISODE times
N_F = 8  # speed,losepacket,application_speed
N_A = 5  # +100,0,-100
A_list = [50, 75, 100, 125, 150]
MAX_BANDWITH = 150  # standardlize to 1
P = [1, 2/3, 1/3]
NN_MODEL = None
# NN_MODEL = './model/v1/nn_model_ep_2.ckpt'
SUMMARY_DIR = './model/v3'
RESULT_DIR = './model/v3.csv'
CC_ADD_RATE = 100
CC_MINUS_RATE = 100


# # we need a good teacher, so the teacher should learn faster than the actor
sess = tf.Session()
actor = Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A)
critic = Critic(sess, n_features=N_F, lr=LR_C)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=10000)
nn_model = NN_MODEL
if nn_model is not None:
    saver.restore(sess, nn_model)
    print("Model restored.")


class RL(CongestionControl):

    def __init__(self):
        super(RL, self).__init__()
        self.USE_CWND = True
        self.send_rate = 1500
        self.cwnd = 120
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
        self.rtt = collections.deque([0]*EPISODE)
        self.eps_start_time = 0
        self.last_inflight = collections.deque([0]*EPISODE)
        self.rtt_min = 100
        self.thr = collections.deque([0]*EPISODE)
        self.last_cwnd = 120

    def estimate_bandwidth(self, cur_time, data):
        event_type = data["event_type"]
        event_time = cur_time

        if cur_time < 0.1:
            self.rtt_min = 100

        if event_type == EVENT_TYPE_DROP:
            self.instant_drop_nums += 1
            self.counter += 1
            self.cur_time = event_time

        if event_type == EVENT_TYPE_FINISHED:
            self.instant_ack_nums += 1
            self.counter += 1
            self.cur_time = event_time

        inflight = data["packet_information_dict"]["Extra"]["inflight"]
        self.last_inflight.append(inflight)
        self.last_inflight.popleft()

        rtt = data["packet_information_dict"]["Latency"] + data["packet_information_dict"]["Pacing_delay"]
        self.rtt.append(rtt)
        self.rtt.popleft()

        if self.rtt_min > rtt:
            self.rtt_min = rtt

        thr = 1500 / 1000 / 1000 / rtt
        self.thr.append(thr)
        self.thr.popleft()

        if self.counter == EPISODE:  # choose action every EPISODE times

            time_interval = cur_time - self.eps_start_time
            # print((self.last_inflight[-1]-self.last_inflight[0])/time_interval, (self.last_send_rate - self.send_rate))
            for i in range(len(self.rtt)):
                self.rtt[i] = self.rtt[i] - self.rtt_min

            thr_use = (self.instant_ack_nums + 0.5 * self.instant_drop_nums) * 1500 / 1000 / 1000 / (
                        np.mean(self.rtt) + self.rtt_min)
            thr_real = (self.instant_ack_nums + self.instant_drop_nums) * 1500 / 1000 / 1000 / self.rtt_min

            # current_state
            state = np.zeros(N_F)
            state[0] = self.cwnd / MAX_BANDWITH
            state[1] = abs(self.cwnd - self.last_cwnd + 1) / MAX_BANDWITH
            state[2] = thr_use / thr_real
            # state[2] = np.mean(self.rtt) * 10
            state[3] = self.instant_drop_nums / EPISODE
            state[4] = (self.last_inflight[-1] - self.last_inflight[0])/(time_interval + 0.0001) / 2500
            state[5] = (self.last_inflight[-1] - self.last_inflight[-5]) / 10
            state[6] = self.last_inflight[-1] / (self.last_inflight[0] + 1)
            state[7] = (self.last_inflight[-1] - self.last_inflight[-3]) / (self.last_inflight[-3] - self.last_inflight[-5] + 0.1)
            # print("%.4f"%state[5], "%.4f"%state[7], "%.4f"%state[8], "%.4f"%state[3],"%.4f"%state[4],"%.4f"%state[5])

            # choose action and explore
            a = actor.choose_action(state)
            # print(a)
            # paket_size Byte, thr MB

            self.reward = 0.2 * self.cwnd /  MAX_BANDWITH + 0.3 * thr_use / thr_real - self.instant_drop_nums / EPISODE - 0.5 * abs(self.cwnd - self.last_cwnd)/MAX_BANDWITH
            self.last_cwnd = self.cwnd
            self.cwnd = A_list[int(a)]
            # print(self.reward, a)

            if self.index > 0:
                td_error = critic.learn(self.last_state, self.reward, state)
                actor.learn(self.last_state, self.last_action, td_error)
            self.last_state = state
            self.last_action = a
            self.reward = 0
            self.index = 1
            self.counter = 0
            self.eps_start_time = cur_time
            self.instant_drop_nums = 0
            self.instant_ack_nums = 0

    def append_input(self, data):
        self._input_list.append(data)

        if data["event_type"] != EVENT_TYPE_TEMP:
            self.cc_trigger(data)
            return {
                "cwnd": self.cwnd,
                "send_rate": self.send_rate
            }
        return None


# Your solution should include packet selection and bandwidth estimator.
# We recommend you to achieve it by inherit the objects we provided and overwritten necessary method.
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

def path_cwd(traces_dir, blocks_dir):
    traces_list = os.listdir(traces_dir)
    for i in range(len(traces_list)):
        traces_list[i] = traces_dir + '/' + traces_list[i]
    blocks_list = os.listdir(blocks_dir)
    for i in range(len(blocks_list)):
        blocks_list[i] = blocks_dir + '/' + blocks_list[i]
    return traces_list, blocks_list

if __name__ == '__main__':
    # set datasets path
    path_dir = "./datasets/"
    # select scenario
    scenario = "scenario_2"
    network_traces, block_traces = path_cwd(path_dir + scenario + "/networks", path_dir + scenario + "/blocks")
    second_block_file = ["../datasets/background_traffic_traces/web.csv"]
    log_packet_file = "./output/packet_log/packet-0.log"

    for i_eps in range(10000):
        qoe_sum = 0
        for network_trace in network_traces:
            # print(network_trace.split('/')[-1])
            my_solution = MySolution()
            emulator = create_emulator(
                block_file=block_traces,
                second_block_file=second_block_file,
                trace_file=network_trace,
                solution=my_solution,
                ENABLE_LOG=False
            )
            emulator.run_for_dur(15)
            print("%.2f" % cal_qoe())
            qoe_sum += cal_qoe()
        save_path = saver.save(sess, SUMMARY_DIR + "/nn_model_ep_" + str(i_eps) + ".ckpt")
        with open(RESULT_DIR, 'a', encoding="utf-8") as f:
            info = str(i_eps) + ',' + str(qoe_sum) + '\n'
            f.write(info)
        print("i_eps:", i_eps, " SUM:", qoe_sum)
