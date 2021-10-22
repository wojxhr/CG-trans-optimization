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
tracecount=0


# Your solution should include block selection and bandwidth estimator.
# We recommend you to achieve it by inherit the objects we provided and overwritten necessary method.
class MySolution(BlockSelection, Reno):

    def __init__(self):
        super().__init__()

        # the value of congestion window
        self.cwnd = 40
        self.send_rate = float("inf")
        self.pacing_rate = float("inf")
        self.USE_CWND=True

        # for reno
        self.curr_state = "slow_start"
        self.states = ["slow_start", "congestion_avoidance", "fast_recovery"]

        self.num = 0
        self.last_rtt = 0
        self.thr_list = collections.deque([0] * EPISODE)
        self.timepoint = 0
        self.bandwith = float()

        # current time
        self.cur_time = -1
        self.new_trace_time = float("inf")
        self.instant_drop_nums = 0
        self.min_cwnd = MIN_CWND
        self.max_cwnd = MAX_CWND
        self.rtt_base = float("inf")
        self.rtt = collections.deque([0] * EPISODE)
        self.inflight = collections.deque([0] * EPISODE)
        self.cwnd_list = collections.deque([0] * EPISODE)
        self.cwnd_list.append(self.cwnd)
        self.cwnd_list.popleft()
        self.last_drop_nums = collections.deque([0] * EPISODE)

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
        """
        The part of algorithm to make congestion control, which will be call when sender get an event about acknowledge or lost from reciever.
        See more at https://github.com/AItransCompetition/simple_emulator/tree/master#congestion_control_algorithmpy.
        """
        global tracecount

        event_type = event_info["event_type"]
        event_time = cur_time
        if cur_time < self.new_trace_time:
            # the value of congestion window
            self.cwnd = 40
            self.send_rate = float("inf")
            self.pacing_rate = float("inf")
            self.USE_CWND = True

            # for reno
            self.curr_state = "slow_start"
            self.states = ["slow_start", "congestion_avoidance", "fast_recovery"]

            self.num=0
            self.last_rtt = 0
            self.thr_list = collections.deque([0] * EPISODE)
            self.timepoint=0
            self.bandwith=float()


            # current time
            self.cur_time = -1
            self.new_trace_time = float("inf")
            self.instant_drop_nums = 0
            self.min_cwnd = MIN_CWND
            self.max_cwnd = MAX_CWND
            self.rtt_base = float("inf")
            self.rtt = collections.deque([0] * EPISODE)
            self.inflight = collections.deque([0] * EPISODE)
            self.cwnd_list = collections.deque([0] * EPISODE)
            self.cwnd_list.append(self.cwnd)
            self.cwnd_list.popleft()
            self.last_drop_nums = collections.deque([0] * EPISODE)
            tracecount+=1
        self.new_trace_time = cur_time

        if self.cur_time < event_time:
            # return
            # initial parameters at a new moment
            self.last_cwnd = 0
            self.last_drop_nums.append(self.instant_drop_nums)
            self.last_drop_nums.popleft()
            self.instant_drop_nums = 0

        send_delay = event_info["packet_information_dict"]["Send_delay"]
        latency = event_info["packet_information_dict"]["Latency"]
        pacing_delay = event_info["packet_information_dict"]["Pacing_delay"]
        rtt = latency + pacing_delay + send_delay
        self.rtt.append(rtt)
        self.rtt.popleft()
        if self.rtt_base > rtt:
            self.rtt_base=rtt


        inflight = event_info["packet_information_dict"]["Extra"]["inflight"]
        self.inflight.append(inflight)
        self.inflight.popleft()


        self.last_event_type = event_type
        if self.num == 0:
            self.last_rtt = rtt
            self.num += 1
        else:
            packet_delay = rtt - self.last_rtt
            if packet_delay != 0 and packet_delay > 0.0001 and packet_delay <= 0.015:
                thr = 1/packet_delay
                self.thr_list.append(thr)
                self.thr_list.popleft()
                # print(cur_time, packet_delay, self.thr_list[-1])
            self.last_rtt = rtt

        # if packet is dropped
        if event_type == EVENT_TYPE_DROP:
            # dropping more than one packet at a same time is considered one event of packet loss
            if self.instant_drop_nums > 0:
                self.instant_drop_nums += 1
                with open("./result" + str(tracecount) + '.csv', 'a+') as file:
                    line = str(cur_time) + ',' + str(self.cwnd) + ',' + str(rtt) + ',' + str(inflight) + ',' + str(
                        self.drop_nums) + ',' + str(self.instant_drop_nums) + '\n'
                    file.writelines(line)
                return
            self.instant_drop_nums += 1
            self.curr_state = self.states[2]
            self.cur_time = event_time

        # if packet is acknowledged
        if event_type == EVENT_TYPE_FINISHED:
            if event_time <= self.cur_time:
                return
            self.cur_time = event_time
            self.last_cwnd = self.cwnd

        if self.cur_time-self.timepoint>1:
            self.timepoint+=1

        if self.cur_time-self.timepoint > self.rtt_base:
            self.bandwith=np.mean(self.thr_list)

        # double cwnd in slow_start state
        if self.curr_state == self.states[0]:
            if self.rtt[-1] < self.rtt_base * 1.2:
                self.cwnd = min(self.max_cwnd, self.cwnd + 2)
            elif self.rtt[-1] > self.rtt_base * 2:
                # print(cur_time)
                self.curr_state = self.states[2]

        # increase cwnd linearly in congestion_avoidance state
        elif self.curr_state == self.states[1]:
            if self.cwnd > self.inflight[-1] * 1.5 and self.cwnd == self.cwnd_list[-1]:
                self.cwnd = max(self.min_cwnd, self.inflight[-1])
            elif np.mean(self.rtt) == self.rtt_base:
                self.cwnd = min(self.max_cwnd, self.cwnd + 5)
            elif self.rtt[-1] < self.rtt_base * 1.2:
                self.cwnd = min(self.max_cwnd, self.cwnd + 1)

        # reset threshhold and cwnd in fast_recovery state
        if self.curr_state == self.states[2]:
            if np.mean(self.rtt) > self.rtt_base * 2:
                print(cur_time,1)
                self.cwnd = self.min_cwnd
            if self.rtt[-1] > self.rtt_base * 2:
                print(cur_time,2)
                self.cwnd = max(self.min_cwnd, self.cwnd - 1)
            else:
                print(cur_time,3)
                self.cwnd = max(self.min_cwnd, self.cwnd // 1.01)
            self.curr_state = self.states[1]

        # if inflight > self.rtt_base * self.bandwith and :
        #     self.cwnd=max(self.min_cwnd,self.cwnd-1)
        # if self.cwnd < self.cwnd_list[-1]:
        #     print(cur_time,self.cwnd,self.cwnd_list[-1])


        self.cwnd_list.append(self.cwnd)
        self.cwnd_list.popleft()
        with open("./result" + str(tracecount) + '.csv', 'a+') as file:
            line = str(cur_time) + ',' + str(self.cwnd) + ',' + str(rtt) + ',' + str(inflight) + ',' + str(
                self.drop_nums) + ',' + str(self.instant_drop_nums) +'\n'
            file.writelines(line)

        return {
            "cwnd": self.cwnd,
            "send_rate": self.send_rate,
        }
