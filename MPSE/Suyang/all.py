import numpy  as np
import matplotlib.pyplot as plt
import MPSE.mview as mview
import sys




def main():
   
    fileName = sys.argv[1]
    begin = int(sys.argv[2])
    end = int(sys.argv[3])

    matrices = []
    for i in range(begin, end+1):
        name = fileName + str(i) + '.csv'
        try:
            y = np.genfromtxt(name, delimiter = ',') 
        except:
            print("generate from", name, "failed\n")
            continue
        print(y)
        print(name)
        matrices.append(y)


    mv = mview.basic(matrices, verbose = 2)
    
    with open ('all.csv', 'w') as fd:
        for each in mv.embedding:
            fd.write(",".join([str(each[0]).strip(), str(each[1]).strip(), str(each[2]).strip()]) +"\n")
    mv.plot_embedding()
    input()

main()
