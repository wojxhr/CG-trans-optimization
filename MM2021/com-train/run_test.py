from simple_emulator import SimpleEmulator, create_emulator
# We provided some function of plotting to make you analyze result easily in utils.py
from simple_emulator import analyze_emulator, plot_rate
from simple_emulator import constant
from simple_emulator import cal_qoe
import os
import importlib
import random
from simple_emulator import Reno
import math
from simple_emulator import BlockSelection
import csv
EVENT_TYPE_FINISHED='F'
EVENT_TYPE_DROP='D'
EVENT_TYPE_TEMP='T'
global new_cwnd

def path_cwd(traces_dir, blocks_dir):
    traces_list = os.listdir(traces_dir)
    for i in range(len(traces_list)):
        traces_list[i] = traces_dir + '/' + traces_list[i]
    blocks_list = os.listdir(blocks_dir)
    for i in range(len(blocks_list)):
        blocks_list[i] = blocks_dir + '/' + blocks_list[i]
    return traces_list, blocks_list


def scenario_result(path_dir, scenario):
    qoe_sum = 0
    random.seed(1)

    my_solution = MySolution()

    # select dataset
    network_traces, block_traces = path_cwd(path_dir + scenario + "/networks", path_dir + scenario + "/blocks")

    # The file path of packets' log
    log_packet_file = "output/packet_log/packet-0.log"

    # The first sender will use your solution, while the second sender will send the background traffic
    # Set second_block_file=None if you want to evaluate your solution in situation of single flow
    # Specify ENABLE_LOG to decide whether or not output the log of packets. ENABLE_LOG=True by default.
    # You can get more information about parameters at https://github.com/AItransCompetition/simple_emulator/tree/master#constant
    # The block files for the first sender
    first_block_file = block_traces
    # The block files for the second sender
    second_block_file = [path_dir + "/background_traffic_traces/web.csv"]

    # Create the emulator and evaluate your solution
    for network_trace in network_traces:
        emulator = create_emulator(
            block_file=first_block_file,
            second_block_file=second_block_file,
            trace_file=network_trace,
            solution=my_solution,
            # enable logging packet. You can train faster if ENABLE_LOG=False
            ENABLE_LOG=True
        )
        emulator.run_for_dur(15)
        # emulator.print_debug()

        # print(network_trace.split("/")[-1], "%.2f" % cal_qoe())
        print("%.2f" % cal_qoe())
        qoe_sum += cal_qoe()

    print("%.2f" % qoe_sum)
    return qoe_sum


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

        # for reno
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

        # self.rtt_sum = 0
        # self.packet_num = 0
        # self.rtt_ave = 0

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
        """
        The part of algorithm to make congestion control, which will be call when sender get an event about acknowledge or lost from reciever.
        See more at https://github.com/AItransCompetition/simple_emulator/tree/master#congestion_control_algorithmpy.
        """

        event_type = event_info["event_type"]
        event_time = cur_time
        # print(event_time)

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
            self.curr_state = self.states[2]
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
            # double cwnd in slow_start state
            if self.curr_state == self.states[0]:
                if self.ack_nums == self.cwnd:
                    self.cwnd *= 2
                    self.ack_nums = 0
                # step into congestion_avoidance state due to exceeding threshhold
                if self.cwnd >= self.ssthresh:
                    self.curr_state = self.states[1]

            # increase cwnd linearly in congestion_avoidance state
            elif self.curr_state == self.states[1]:
                if self.ack_nums == self.cwnd:
                    self.cwnd += 1
                    self.ack_nums = 0

        # reset threshhold and cwnd in fast_recovery state
        if self.curr_state == self.states[2]:
            self.ssthresh = max(self.cwnd // 2, 1)
            self.cwnd = self.ssthresh
            self.curr_state = self.states[1]

        # set cwnd or sending rate in sender
        # latency = event_info["packet_information_dict"]["Latency"]
        # pacing_delay = event_info["packet_information_dict"]["Pacing_delay"]
        # rtt = latency + pacing_delay
        # self.rtt_sum += rtt
        # if self.packet_num == 0:
        #     self.rtt_ave = rtt
        # self.packet_num += 1
        # print(self.rtt_ave)

        # # print(rtt)
        # if rtt > 0.08:
        #     self.cwnd = max(self.cwnd / 1.02, 1)
        # elif rtt > 0.06:
        #     self.cwnd = max(self.cwnd / 1.01, 1)
        # else:
        #     self.cwnd = 75
        # if rtt > self.rtt_ave + 0.01:
        #     self.cwnd = 100000
        # else:
        #     self.cwnd = 75
        self.cwnd = new_cwnd

        return {
            "cwnd" : self.cwnd,
            "send_rate" : self.send_rate,
        }


if __name__ == '__main__':
    # Select the solution file
    # solution_file = 'solution_ours.reno.method_3'
    # solution_file = 'solution_ours.reno.method_2'
    # solution_file = 'solution_ours.rl_Tensorflow.method_1.solution'
    # solution_file = 'solution_ours.rl_Tensorflow.method_2.solution'

    # set datasets path
    path_dir = "/home/ftt/PycharmProjects/MM2021/com-train/datasets/"

    # select scenario
    scenario = "scenario_"

    # result_path = "/home/ftt/PycharmProjects/MM2021/com-train/analysis/test.csv"
    for j in range(80,180):
        with open('./analysis/test.csv', 'a+') as csvfile:
            line = str(j)
            new_cwnd = j
            qoe_all = 0
            for i in range(1, 4):
                scenario = "scenario_" + str(i)
                qoe = scenario_result(path_dir, scenario)
                line = line + ',' + str(qoe)
                qoe_all += qoe
            line = line + ',' + str(qoe_all)
            line = line + '\n'
            csvfile.writelines(line)


