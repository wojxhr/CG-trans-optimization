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
        # the value of sending rate
        self.send_rate = float("inf")
        # the value of pacing rate
        self.pacing_rate = float("inf")
        # use cwnd
        self.USE_CWND=True

        self.length = 50
        self.rtt_list = np.zeros([self.length])
        self.packet_num = 0
        self.rtt_ave = 0
        self.last_cur_time = 0

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

        event_type = event_info["event_type"]
        event_time = cur_time

        latency = event_info["packet_information_dict"]["Latency"]
        pacing_delay = event_info["packet_information_dict"]["Pacing_delay"]
        rtt = latency + pacing_delay

        if cur_time < self.last_cur_time:
            self.cwnd = 1
            self.send_rate = float("inf")
            self.USE_CWND = True

            self.length = 50
            self.rtt_list = np.zeros([self.length])
            self.packet_num = 0
            self.rtt_ave = 0
            self.last_cur_time = 0

        self.last_cur_time = cur_time

        # print(rtt)
        if rtt > 1.5*self.rtt_ave:
            self.cwnd = max(80, self.cwnd * 0.95)
        elif rtt > 1.2*self.rtt_ave:
            self.cwnd = max(80, self.cwnd * 0.99)
        elif rtt < 0.6 * self.rtt_ave:
            self.cwnd = min(140, self.cwnd * 1.02)
        elif rtt < 0.8 * self.rtt_ave:
            self.cwnd = min(140, self.cwnd * 1.05)
        else:
            self.cwnd = 120

        # self.cwnd = 50
        self.rtt_list = np.roll(self.rtt_list, -1, axis=0)
        self.rtt_list[-1] = rtt
        # print(self.rtt_list)
        self.packet_num = min((self.packet_num + 1), self.length)
        self.rtt_ave = np.sum(self.rtt_list) / self.packet_num
        # print(self.rtt_ave)

        self.send_rate = 2500
        self.cwnd = 25


        return {
            "cwnd" : self.cwnd,
            "send_rate" : self.send_rate,
        }