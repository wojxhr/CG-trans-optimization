from simple_emulator import SimpleEmulator, create_emulator
# We provided some function of plotting to make you analyze result easily in utils.py
from simple_emulator import analyze_emulator, plot_rate
from simple_emulator import constant
from simple_emulator import cal_qoe
import os
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


def scenario_result(solution_file, path_dir, scenario):
    qoe_sum = 0
    random.seed(1)

    solution = importlib.import_module(solution_file)
    my_solution = solution.MySolution()

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

        # print(network_trace.split("/")[-1], cal_qoe())
        print("%.2f" % cal_qoe())
        qoe_sum += cal_qoe()

    print("%.2f" % qoe_sum)


# for scenario_1~scenario_3
if __name__ == '__main__':
    # Select the solution file
    # solution_file = 'solution_demos.reno.solution'
    # solution_file = 'solution_demos.reno.5_8'
    # solution_file = 'solution_demos.reno.test'
    # solution_file = 'solution_demos.rl_tensorflow.solution'
    solution_file = 'solution'

    # set datasets path
    path_dir = "/home/ftt/PycharmProjects/MM2021/com-train/datasets/"

    # select scenario
    scenario = "scenario_1"

    # print result
    scenario_result(solution_file, path_dir, scenario)
