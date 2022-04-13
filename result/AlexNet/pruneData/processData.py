count = 0
data = []
if __name__ == '__main__':
    with open("./sparisty.txt", 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            data.append(round(1.0 - float(line), 4))
            count += 1
            if count >= 8:
                break
    print(data)