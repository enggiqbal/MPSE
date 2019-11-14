import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


points_data=pd.read_csv("sample.csv", header=None)
points_data=points_data[0:100]
#points_data[2]=points_data[2].round(1)
#points_data[1]=points_data[1].round(1)
#points_data[0]=points_data[0].round(1)
points_data=points_data.values
'''
P1=np.array([[1,0,0], [0,0,0],[0,0,1]], dtype=float)
P2=np.array([[1,0,0], [0,1,0],[0,0,0]], dtype=float)
P3=np.array([[0,0,0], [0,1,0],[0,0,1]], dtype=float)

ps1=get2Dproject(points_data, P1,[True, True, False])
ps2=get2Dproject(points_data, P2,[False, True, True])
ps3=get2Dproject(points_data, P3,[True, False, True])
'''
points_data=points_data+20
points_data=points_data/4
points= points_data.T[0:2].T



fig = plt.figure()
ax = plt.axes()
ax.grid(False)
plt.axis('off')
ax.scatter(points.T[0], points.T[1],  c='green', marker="x" ,  cmap='Greens');
plt.savefig(filename)



dpi=192
fig = plt.figure(frameon=False)
ax = plt.axes()
fig.suptitle('10 x 10 inch ', fontsize=20)
#plt.xlim(0, 10)
plt.xticks(np.arange(0,10, step=0.2))
plt.yticks(np.arange(0,10, step=0.2))
#plt.ylim(0, 10)
#plt.figure(figsize=(10 , 10), dpi=100 )
fig.set_size_inches(10,10)
for i  in range(0, len(points)):
    x = points[i][0]
    y = points[i][1]
    z= round( points_data[i][2], 1)
    ax.scatter(x, y, marker='.', color='red')
#    ax.text(x-0.05, y-0.05, str(  round(x, 1)) + ","+ str(round(y,1)) , fontsize=9)

#fig.show()
plt.savefig("topview5.png" ,dpi=100)



np.savetxt("data.csv",points_data,delimiter=",", fmt='%.1f')


'''

draw2d(p1, "projection01.png")

draw2d(ps1, "projection1.png")
draw2d(ps2, "projection2.png")
draw2d(ps3, "projection3.png")



pos=pd.read_csv("123_12p2_pos.csv")
pos=pos.sample(100)
pos=pos.values
pos.to_csv("sample.csv",index=False)
jsdata ="var points="+ str(  pos.tolist()) + ";"
f=open("smaple.js","w")
f.write(jsdata);
f.close()

'''
