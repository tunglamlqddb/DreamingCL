import matplotlib
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np

res_path = 'outputs'

def get_data(path):
    with open(path,'r') as f:
        data = f.readlines()
        data = np.array(data)
    return data

def line_plot(res):
    x = np.arange(res[0].shape[0])

    plt.plot(x, res[0], label = "ReBN")
    plt.plot(x, res[1], label = "No-ReBN")
    plt.xlabel("Epoch")
    plt.ylabel("Val Loss")
    plt.title('Performance of Generator w and w/o ReBN')
    plt.legend()
    plt.show()

paths = [res_path+'/rebn.txt', res_path+'/no-rebn.txt']
res = []
for path in paths:
    res.append(get_data(path))
line_plot(res)
