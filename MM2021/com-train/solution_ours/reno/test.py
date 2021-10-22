"""
This demo aims to help player running system quickly by using the pypi library simple-emualtor https://pypi.org/project/simple-emulator/.
"""
from simple_emulator import CongestionControl

# We provided a simple algorithms about block selection to help you being familiar with this competition.
# In this example, it will select the block according to block's created time first and radio of rest life time to deadline secondly.
from simple_emulator import BlockSelection

# We provided some simple algorithms about congestion control to help you being familiar with this competition.
# Like Reno and an example about reinforcement learning implemented by tensorflow
from simple_emulator import Reno
import math
import numpy as np
import collections

EPISODE = 25
MAX_CWND = 170
MIN_CWND = 40
EVENT_TYPE_FINISHED='F'
EVENT_TYPE_DROP='D'
EVENT_TYPE_TEMP='T'

# Your solution should include block selection and bandwidth estimator.
# We recommend you to achieve it by inherit the objects we provided and overwritten necessary method.
class MySolution(BlockSelection, Reno):

    def __init__(self):
        super().__init__()
        # base parameters in CongestionControl

        # the value of congestion window
        self.cwnd = 1
        self.cwnd2 = 1
        self.cwnd3 = 1
        self.send_rate = float("inf")
        self.pacing_rate = float("inf")
        self.USE_CWND=True

        # for reno
        self.ssthresh = float("inf")
        self.curr_state = "slow_start"
        self.states = ["slow_start", "congestion_avoidance", "fast_recovery"]
        self.drop_nums = 0
        self.ack_nums = 0

        # current time
        self.cur_time = -1
        self.last_cwnd = 0
        self.instant_drop_nums = 0

        self.min_cwnd = MIN_CWND
        self.max_cwnd = MAX_CWND
        self.rtt_base = 100
        self.rtt = collections.deque([0] * EPISODE)
        self.index = 1
        self.inflight = collections.deque([0] * EPISODE)
        self.cwnd_list = collections.deque([0] * EPISODE)
        self.cwnd_list.append(self.cwnd)
        self.cwnd_list.popleft()

        # 队列的每个元素是一个字典，字典的key是cur_time，value是packet_id
        self.bbr_ack_time = [100]*200

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
            best_block_ddl_ratio = best_block_rest / (best_block_ddl_rest + 0.0001)
            block_ddl_radio = block_rest /  (block_ddl_rest + 0.0001)
            best_block_rest_pkt = math.ceil(best_block_rest / 1480)
            block_rest_pkt = math.ceil(block_rest / 1480)

            # if best_block is miss ddl
            if (cur_time - best_block_create_time) > best_block.block_info["Deadline"]:
                return True
            # if block is miss ddl
            if (cur_time - cur_block_create_time) > block.block_info["Deadline"]:
                return False
            # retrans-packet is first
            if block.retrans == True:
                return True
            if best_block.retrans == True:
                return False

            # block_rest is less
            if best_block_rest_ratio * best_block_rest_pkt * 2 <= block_rest_ratio * block_rest_pkt:
                return False
            # block_ddl_rest is less
            if best_block_ddl_rest / best_block_rest_pkt >= block_ddl_rest / block_rest_pkt * 2:
                return False
            # all information
            if best_block_ddl_ratio * best_block_rest_ratio * (0.01 + 0.5 * best_block_priority) > block_ddl_radio * \
                    block_rest_ratio * (0.01 + 0.5 * cur_block_priority):
                return True
            else:
                return False

        best_block_idx = -1
        best_block= None
        for idx, item in enumerate(block_queue):
            if best_block is None or is_better(item) :
                best_block_idx = idx
                best_block = item

        return best_block_idx

    def on_packet_sent(self, cur_time):
        """
        The part of solution to update the states of the algorithm when sender need to send pacekt.
        """
        return super().on_packet_sent(cur_time)

    def cc_trigger(self, cur_time, event_info):
        """
        The part of algorithm to make congestion control, which will be call when sender get an event about acknowledge or lost from reciever.
        See more at https://github.com/AItransCompetition/simple_emulator/tree/master#congestion_control_algorithmpy.
        """
        ack_time_difference=100

        event_type = event_info["event_type"]
        event_time = cur_time

        latency = event_info["packet_information_dict"]["Latency"]
        pacing_delay = event_info["packet_information_dict"]["Pacing_delay"]
        rtt = latency + pacing_delay
        self.rtt.append(rtt)
        self.rtt.popleft()

        inflight = event_info["packet_information_dict"]["Extra"]["inflight"]
        self.inflight.append(inflight)
        self.inflight.popleft()

        if self.rtt_base > rtt:
            self.rtt_base = rtt

        if self.cur_time < event_time:
            # initial parameters at a new moment
            self.last_cwnd = 0
            self.instant_drop_nums = 0

        # if packet is dropped
        if event_type == EVENT_TYPE_DROP:
            self.drop_nums += 1
            # dropping more than one packet at a same time is considered one event of packet loss
            if self.instant_drop_nums > 0:
                return
            self.instant_drop_nums += 1
            # step into fast recovery state
            self.curr_state = self.states[2]
            # self.drop_nums += 1
            # clear acknowledgement count
            self.ack_nums = 0
            # Ref 1 : For ensuring the event type, drop or ack?
            self.cur_time = event_time
            if self.last_cwnd > 0 and self.last_cwnd != self.cwnd:
                # rollback to the old value of cwnd caused by acknowledgment first
                self.cwnd = self.last_cwnd
                self.last_cwnd = 0

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
                if self.ack_nums >= self.cwnd:

                    self.cwnd = min(self.max_cwnd, self.cwnd * 6)
                    self.ack_nums = self.ack_nums

            # increase cwnd linearly in congestion_avoidance state
            elif self.curr_state == self.states[1]:
                # if self.rtt[-1] > self.rtt_base:
                #     self.cwnd = max(self.min_cwnd, self.cwnd // 1.001)
                if self.ack_nums >= self.cwnd:
                    self.cwnd = min(self.max_cwnd, self.cwnd + 8)
                    self.ack_nums = 0


        # reset threshhold and cwnd in fast_recovery state
        if self.curr_state == self.states[2]:
            self.ssthresh = max(self.min_cwnd, self.cwnd // 1.0078)
            self.cwnd = self.ssthresh
            self.curr_state = self.states[1]

        if self.index % 5 == 0:
            if np.mean(self.rtt) == self.rtt_base:
                self.cwnd = min(self.max_cwnd, self.cwnd + 10)
            elif self.cwnd_list[-1] - self.cwnd_list[-2] - self.inflight[-1] + self.inflight[-2] > 0:
                self.cwnd = min(self.max_cwnd, self.cwnd * 1.5)

            if np.mean(self.rtt) > self.rtt_base * 2:
                self.cwnd = MIN_CWND

        self.index += 1
        self.cwnd_list.append(self.cwnd)
        self.cwnd_list.popleft()




        with open("./result.csv", 'a+') as file:
            line = str(cur_time) + ',' + str(self.cwnd) + ',' + str(rtt) + ',' + str(inflight) + ',' + str(ack_time_difference) + '\n'
            file.writelines(line)

        return {
            "cwnd": self.cwnd,
            "send_rate": self.send_rate,
        }