import importlib

import simple_emulator
from simple_emulator.objects.link import Link
from simple_emulator.objects.sender import Sender

def get_trace(trace_file):
    """init the "trace_list" according to the trace file."""
    trace_list = []
    trace_cols = ("time", "bandwith", "loss_rate", "delay")
    with open(trace_file, "r") as f:
        for line in f.readlines():
            trace_list.append(list(
                map(lambda x: float(x), line.split(","))
            ))
            if len(trace_list[-1]) != len(trace_cols):
                raise ValueError("Trace file error!\nPlease check its format like : {0}".format(self.trace_cols))

    if len(trace_list) == 0:
        raise ValueError("Trace file error!\nThere is no data in the file!")

    return trace_list



if __name__ == '__main__':

    block_traces = ["datasets/scenario_2/blocks/block_video.csv", "datasets/scenario_2/blocks/block_audio.csv"]
    network_trace = "datasets/scenario_2/networks/traces_7.txt"
    solution_file = 'aitrans_solution'
    first_block_file = block_traces
    second_block_file = ["datasets/background_traffic_traces/web.csv"]

    list = get_trace(network_trace)
    queue = 55
    links=Link(list,queue)
    # features = []
    # solution=importlib.import_module(solution_file)
    # my_solution = solution.MySolution()
    # sender=Sender(links,0,features,history_len=1,solution=my_solution)

    print(links)
