import re
import csv
import numpy as np
from matplotlib import pyplot as plt

predict_time = []
layers_name = []
original_time = []


def collectData(input_file, output_init='/homework/lv/Time analysis and Optimization Based on CNN accelerator/result/AlexNet/Data2.txt'):
    input_f = open(input_file, 'r')
    output_conv = open(output_init + '_conv.txt', 'w')
    conv_signs = ['conv', 'incept', 'res', 'cccp']
    for line in input_f:
        if len(line.strip()) == 0: continue
        layer_name = line.split()[0].lower()

        if any(conv in layer_name for conv in conv_signs):
            res = []
            items = line.split(':')
            # print(items)
            res += re.findall(r"[-+]?\d*\.\d+|\d+", items[0])[-4:]  # Output
            res += re.findall(r"[-+]?\d*\.\d+|\d+", items[1])  # Filters
            res += re.findall(r"[-+]?\d*\.\d+|\d+", items[2])  # Padding
            res += re.findall(r"[-+]?\d*\.\d+|\d+", items[3])  # strides
            res += re.findall(r"[-+]?\d*\.\d+|\d+", items[5])  # Inputs
            res += re.findall(r"[-+]?\d*\.\d+|\d+", items[6])  # Input_nonzero
            res += re.findall(r"[-+]?\d*\.\d+|\d+", items[7])  # filter_nonzero
            res += re.findall(r"[-+]?\d*\.\d+|\d+", items[8])  # Runtime
            original_time.append(float(res[-1]))
            output_conv.write(', '.join(res) + '\n')

    input_f.close()
    output_conv.close()


def parse_results(input_file, coeffi):
    input_f = open(input_file, 'r')
    conv_signs = ['conv', 'res', 'cccp']
    for line in input_f:
        if len(line.strip()) == 0: continue
        if 'json' in line and 'Network' in line.split()[0]:
            print("\n%s" % line.split('/')[-1])
        layer_name = line.split()[0].lower()
        if any(conv in layer_name for conv in conv_signs):
            res = []
            items = line.split(':')
            res += re.findall(r"[-+]?\d*\.\d+|\d+", items[0])[-4:]  # Output
            res += re.findall(r"[-+]?\d*\.\d+|\d+", items[1])  # Filters
            res += re.findall(r"[-+]?\d*\.\d+|\d+", items[2])  # Padding
            res += re.findall(r"[-+]?\d*\.\d+|\d+", items[3])  # strides
            res += re.findall(r"[-+]?\d*\.\d+|\d+", items[5])  # Inputs
            res += re.findall(r"[-+]?\d*\.\d+|\d+", items[6])  # Input_nonzero
            res += re.findall(r"[-+]?\d*\.\d+|\d+", items[7])  # filter_nonzero
            # print(res)
            input = map(float, res)
            pre_runtime = predict_runtime('conv', input, coeffi)
            print("%s\t%.3f" % (layer_name, pre_runtime))
            layers_name.append(layer_name)
            predict_time.append(pre_runtime)
    input_f.close()


def predict_runtime(type, input, coeffi):
    input = list(input)
    if type == 'conv':
        input_1 = input[0:3] + input[5:8] + input[13:14]
        input_2 = []
        for i in range(len(input_1)):
            for j in range(i, len(input_1)):
                input_2.append(input_1[i] * input_1[j])
        input_3 = []
        for i in range(len(input_1)):
            for j in range(i, len(input_1)):
                for k in range(j, len(input_1)):
                    input_3.append(input_1[i] * input_1[j] * input_1[k])
        p = input
        input_others = [p[1] * p[2] * p[3] * p[4] * p[5] * p[6],  # output pixels
                        p[12] * p[1] * p[2] * p[3] * p[4] * p[5] * p[6],
                        p[12] * p[1] * p[2] * p[4] * p[5] * p[6],
                        p[0] * p[1] * p[2] * p[3],  # output data
                        p[4] * p[5] * p[6] * p[7],  # filter data
                        p[12] * p[13] * p[14] * p[15],  # input data
                        p[12] * p[14] * p[15],  # input data
                        p[12] * p[13] * p[15],
                        p[16],
                        p[17],
                        p[16] * p[17]]  # input data
        input_runtime = input_1 + input_2 + input_3 + input_others + [1]  # 1 is the intercept
        runtime = sum(np.array(input_runtime) * np.array(coeffi[type, 'runtime']))
        return runtime


def parse_coeff(coeffi):
    with open('/homework/lv/Time analysis and Optimization Based on CNN accelerator/result/AlexNet/coeff_conv.txt', 'r') as f:
        res = csv.reader(f)
        coeffi[('conv', 'runtime')] = list(map(float, next(res)))

    return coeffi


if __name__ == '__main__':
    collectData("/homework/lv/Time analysis and Optimization Based on CNN accelerator/result/AlexNet/AlexNetData.txt")
    coeffi = {}
    parse_coeff(coeffi)
    parse_results("/homework/lv/Time analysis and Optimization Based on CNN accelerator/result/AlexNet/Data.txt", coeffi)

    x = layers_name
    y = original_time
    print(y)
    z = predict_time
    total_y = 0
    total_z = 0
    for i in range(len(x)):
        total_y += y[i]
        total_z += z[i]
        y[i] = round(y[i], 2)
        z[i] = round(z[i], 2)
    print("total_original_time: ", total_y)
    print("\ntotal_predict_time: ", total_z)

    # 创建分组柱状图，需要自己控制x轴坐标
    xticks = np.arange(len(x))

    fig, ax = plt.subplots(figsize=(10, 7))
    plt.grid(axis='y', linestyle='--')
    ax.bar(xticks, y, width=0.25, label="original_time", color="red")
    ax.bar(xticks + 0.25, z, width=0.25, label="predict_time", color="blue")
    # 为每个条形图添加数值标签
    for i, v in enumerate(y):
        ax.text(i - 0.15, v + 0.15, v, ha='center', fontsize=10)
    for i, v in enumerate(z):
        ax.text(i + 0.35, v + 0.15, v, ha='center', fontsize=10)
    ax.set_title("AlexNet layers' runtime", fontsize=15)
    ax.set_xlabel("layers_name")
    ax.set_ylabel("layers_runtime / s")
    ax.legend()

    # 最后调整x轴标签的位置
    ax.set_xticks(xticks + 0.125)
    ax.set_xticklabels(x)
    plt.savefig("/homework/lv/time_performance_optimization_for_CNN/result/AlexNet/time_comparison_chart2.pdf")
    plt.show()
