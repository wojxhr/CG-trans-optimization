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

EVENT_TYPE_FINISHED = 'F'
EVENT_TYPE_DROP = 'D'
EVENT_TYPE_TEMP = 'T'

PHASE_COUNT=0
G=[1.25,0.75,1,1,1,1,1,1]
MAX_RATE=1000
MIN_RATE=40



# Your solution should include block selection and bandwidth estimator.
# We recommend you to achieve it by inherit the objects we provided and overwritten necessary method.
class MySolution(BlockSelection, Reno):

    def __init__(self):
        super().__init__()
        # base parameters in CongestionControl

        # the value of congestion window
        self.cwnd = 1
        # the value of sending rate
        self.send_rate = MIN_RATE
        # the value of pacing rate
        self.pacing_rate = float("inf")
        # use cwnd
        self.USE_CWND = False

        # for BBR
        self.curr_state = "startup"

        # 我认为probe_RTT阶段在当前数据集下最好去除，因为其影响了一段时间的发送速率，我们最终考虑的是到达的包越多越好
        self.states = ["startup", "drain", "probe_BW","probe_RTT"]
        self.min_rtt=float("inf")
        self.min_rtt_stamp=0
        self.probe_rtt_stamp=-1
        self.bw=float("inf")
        self.BltBw=float("-inf")
        self.timepoint=0
        self.epoch=0

        # the number of lost packets
        self.drop_nums = 0
        # the number of acknowledgement packets
        self.ack_nums = 0

        # current time
        self.cur_time = -1
        self.last_cwnd = 0
        self.instant_drop_nums = 0



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

            if block.retrans == True:
                return True
            if best_block.retrans == True:
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
        The part of solution to update the states of the algorithm when sender need to send pacekt.
        """
        return super().on_packet_sent(cur_time)

    def cc_trigger(self, cur_time, event_info):
        """
        The part of algorithm to make congestion control, which will be call when sender get an event about acknowledge or lost from reciever.
        See more at https://github.com/AItransCompetition/simple_emulator/tree/master#congestion_control_algorithmpy.
        """

        # BBR congest control
        def estimate_bw(inflight,rtt):
            return float(inflight * 1500 / rtt / 1024 / 1024)

        global PHASE_COUNT
        event_type = event_info["event_type"]
        event_time = cur_time
        latency = event_info["packet_information_dict"]["Latency"]
        pacing_delay = event_info["packet_information_dict"]["Pacing_delay"]
        rtt = latency + pacing_delay
        inflight = event_info["packet_information_dict"]["Extra"]["inflight"]

        if self.cur_time < event_time:
            # initial parameters at a new moment
            self.last_cwnd = 0
            self.instant_drop_nums = 0

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

            # 每秒更新采样
            if self.cur_time-self.timepoint >= 1:
                self.min_rtt=float("inf")
                self.min_rtt_stamp=float("inf")
                self.BltBw=float("-inf")
                self.timepoint=self.cur_time


            self.bw = estimate_bw(inflight, rtt)
            print(self.bw)
            if self.bw > self.BltBw:
                self.BltBw=self.bw
            if rtt < self.min_rtt:
                self.min_rtt=rtt
                self.min_rtt_stamp=self.cur_time

            # 启动阶段
            if self.curr_state == self.states[0]:
                self.send_rate=min(10 * self.BltBw,MAX_RATE)
                self.send_rate=max(self.send_rate,MIN_RATE)
                if self.bw < self.BltBw:
                    PHASE_COUNT+=1
                if PHASE_COUNT >= 5:
                    self.curr_state=self.states[1]
                    PHASE_COUNT=0
            # 排空阶段
            if self.curr_state == self.states[1]:
                self.send_rate=max(MIN_RATE, self.send_rate/1.05)
                if inflight < self.BltBw * self.min_rtt:
                    self.curr_state=self.states[2]
            # probe_BW阶段
            if self.curr_state == self.states[2]:
                self.send_rate=min(self.BltBw * G[self.epoch],MAX_RATE)
                self.epoch+=1
                if self.epoch ==8:
                    self.epoch=0
            # probe_RTT阶段,在比赛环境下考虑何时进入探测阶段？
            if self.curr_state == self.states[3]:
                pass

        with open("./resultBBR.csv", 'a+') as file:
            line = str(cur_time) + ',' + str(int(self.send_rate * 1024 * 1024 / 1500))+','+str(rtt)+','+str(inflight)+','+str(self.drop_nums)+'\n'
            file.writelines(line)
        return {
            "cwnd": self.cwnd,
            "send_rate": int(self.send_rate * 1024 * 1024 / 1500),
        }
