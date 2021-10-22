"""
This demo aims to help player running system quickly by using the pypi library DTP-Emualtor https://pypi.org/project/DTP-Emulator/.
"""
from simple_emulator import PccEmulator, CongestionControl
from simple_emulator import Packet_selection
from simple_emulator import Reno
# from simple_emulator import RL
from simple_emulator import analyze_pcc_emulator, plot_rate
from simple_emulator import constant
from simple_emulator import cal_qoe
import math
import os

EVENT_TYPE_FINISHED = 'F'
EVENT_TYPE_DROP = 'D'
EVENT_TYPE_TEMP = 'T'


# Your solution should include packet selection and congestion control.
# So, we recommend you to achieve it by inherit the objects we provided and overwritten necessary method.
class MySolution(Packet_selection, Reno):

    def __init__(self):
        super().__init__()
        # base parameters in CongestionControl
        # the data appended in function "append_input"
        self._input_list = []
        # the value of crowded window
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
        self.packet_list = []

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
            if best_block_ddl_ratio * best_block_rest_ratio * (0.01 + 0.5 * best_block_priority) > block_ddl_radio * \
                    block_rest_ratio * (0.01 + 0.5 * packet_block_priority):
                return True
            else:
                return False

        best_packet_idx = -1
        best_packet = None
        for idx, item in enumerate(packet_queue):
            if best_packet is None or is_better(item):
                best_packet_idx = idx
                best_packet = item

        return best_packet_idx

    def make_decision(self, cur_time):
        """
        The part of algorithm to make congestion control, which will be call when sender need to send pacekt.
        See more at https://github.com/AItransCompetition/simple_emulator/tree/master#congestion_control_algorithmpy.
        """
        return super().make_decision(cur_time)

    def cc_trigger(self, data):
        event_type = data["event_type"]
        event_time = data["event_time"]

        # see Algorithm design problem in QA section of the official website
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
                    self.cwnd *= 9
                    self.ack_nums = 0
                # step into congestion_avoidance state due to exceeding threshhold
                if self.cwnd >= self.ssthresh:
                    self.curr_state = self.states[1]

            # increase cwnd linearly in congestion_avoidance state
            elif self.curr_state == self.states[1]:
                if self.ack_nums == self.cwnd:
                    self.cwnd += 7
                    self.ack_nums = 0

        # reset threshhold and cwnd in fast_recovery state
        if self.curr_state == self.states[2]:
            self.ssthresh = max(self.cwnd // 1.01, 1)
            self.cwnd = self.ssthresh
            self.curr_state = self.states[1]

    def append_input(self, data):
        """
        The part of algorithm to make congestion control, which will be call when sender get an event about acknowledge or lost from reciever.
        See more at https://github.com/AItransCompetition/simple_emulator/tree/master#congestion_control_algorithmpy.
        """
        # add new data to history data
        self._input_list.append(data)

        # only handle acknowledge and lost packet
        if data["event_type"] != EVENT_TYPE_TEMP:
            # specify congestion control algorithm
            self.cc_trigger(data)
            # set cwnd or sending rate in sender
            return {
                "cwnd" : self.cwnd,
                "send_rate" : self.send_rate
            }
        return None


def path_cwd(traces_dir, blocks_dir):
    traces_list = os.listdir(traces_dir)
    for i in range(len(traces_list)):
        traces_list[i] = traces_dir + '/' + traces_list[i]
    blocks_list = os.listdir(blocks_dir)
    for i in range(len(blocks_list)):
        blocks_list[i] = blocks_dir + '/' + blocks_list[i]
    return traces_list, blocks_list


if __name__ == '__main__':
    traces_list, blocks_list = path_cwd("train_test/day_3/networks", "train_test/day_3/blocks")
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
