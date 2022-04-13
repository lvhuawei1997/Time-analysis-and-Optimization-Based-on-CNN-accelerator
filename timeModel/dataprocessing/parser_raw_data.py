import re


def collectData(input_file, output_init='/homework/lv/Time analysis and Optimization Based on CNN accelerator/result/AlexNet/res'):
    input_f = open(input_file, 'r')
    output_conv = open(output_init + '_conv.txt', 'w')
    conv_signs = ['conv', 'incept', 'res', 'cccp']
    for line in input_f:
        if len(line.strip()) == 0: continue
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
            res += re.findall(r"[-+]?\d*\.\d+|\d+", items[8])  # Runtime
            output_conv.write(', '.join(res) + '\n')

    input_f.close()
    output_conv.close()


if __name__ == '__main__':
    collectData("/homework/lv/Time analysis and Optimization Based on CNN accelerator/result/AlexNet/AlexNetData.txt")

