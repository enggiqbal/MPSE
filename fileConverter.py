import sys

def strip_data(line)
    for each in line:
        each = each.strip()
    return line

def read(fileName):
    data = []
    with open(fileName, 'r') as fd:
        for line in fd:
            each = line.strip().split()
            each = strip_data(each)
            data.append(each)
    return data


def write(data, fileName):
    with open(fileName, 'w') as fd:
        for line in data:
            fd.write(",".join(line)+"\n")

        
        
fileName = sys.argv[1]
begin = int(sys.argv[2])
end = int(sys.argv[3])

for i in range(begin, end+1):
    inputFile = fileName + str(i)
    outputFile = fileName + str(i) + '.csv'
    try:
        data = read(inputFile)
    except:
        print(inputFile, "read failed")
        continue
    print(outputFile)
    write(data, outputFile)

