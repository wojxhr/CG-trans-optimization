from simple_emulator import SimpleEmulator, create_emulator
# We provided some function of plotting to make you analyze result easily in utils.py
from simple_emulator import analyze_emulator, plot_rate
from simple_emulator import constant
from simple_emulator import cal_qoe
import os, inspect
import importlib
import random

def path_cwd(traces_dir, blocks_dir):
    traces_list = os.listdir(traces_dir)
    for i in range(len(traces_list)):
        traces_list[i] = traces_dir + '/' + traces_list[i]
    blocks_list = os.listdir(blocks_dir)
    for i in range(len(blocks_list)):
        blocks_list[i] = blocks_dir + '/' + blocks_list[i]
    return traces_list, blocks_list


# for scenario_1~scenario_3
if __name__ == '__main__':
    # Select the solution file
    solution_file = 'solution'
    solution = importlib.import_module(solution_file)
    my_solution = solution.MySolution()

    # set datasets
    path_dir = "/home/ftt/PycharmProjects/MM2021/com-train/datasets/"
    scenario = "scenario_2"
    network_traces, block_traces = path_cwd(path_dir + scenario + "/networks", path_dir + scenario + "/blocks")
    second_block_file = [path_dir + "/background_traffic_traces/web.csv"]
    log_packet_file = "output/packet_log/packet-0.log"

    qoe_sum = 0
    for network_trace in network_traces:
        # print(network_trace.split('/')[-1])
        emulator = create_emulator(
            block_file=block_traces,
            second_block_file=second_block_file,
            trace_file=network_trace,
            solution=my_solution,
            # enable logging packet. You can train faster if ENABLE_LOG=False
            ENABLE_LOG=False
        )
        emulator.run_for_dur(15)
        # emulator.print_debug()
        # analyze_pcc_emulator(log_packet_file, file_range="all")
        # plot_rate(log_packet_file, trace_file="traces/trace.txt", file_range="all")
        # print(network_trace.split("/")[-1], "%.2f" % cal_qoe())
        print("%.2f" % cal_qoe())
        qoe_sum += cal_qoe()

    print("%.2f" % qoe_sum)
