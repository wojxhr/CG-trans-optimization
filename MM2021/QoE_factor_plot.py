import matplotlib.pyplot as plt
import json
import numpy as np

plt.style.use("ggplot")

def QoE_plot(label,file):
    trace_count=1
    trace_label=[]
    H_data=[]
    M_data=[]
    L_data=[]
    with open(file,"r") as f:
        for line in f.readlines():
            data = json.loads(line.replace("'", '"'))
            trace_label.append("trace"+str(trace_count))
            trace_count+=1

            H_data.append(data['H_'+label])
            M_data.append(data['M_'+label])
            L_data.append(data['L_'+label])

    xticks = np.arange(len(trace_label))
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.bar(xticks, H_data, width=0.25, label='H_'+label, color="red")
    ax.bar(xticks + 0.25, M_data, width=0.25, label='M_'+label, color="blue")
    ax.bar(xticks + 0.5, L_data, width=0.25, label='L_'+label, color="green")

    ax.set_title("All trace_"+label, fontsize=15)
    ax.set_xlabel("Traces")
    ax.set_ylabel(label)
    ax.legend()

    ax.set_xticks(xticks + 0.25)
    ax.set_xticklabels(trace_label)

    plt.show()

if __name__ == '__main__':
    file="./avg_delay_info.log"
    QoE_plot('avg_delay',file)