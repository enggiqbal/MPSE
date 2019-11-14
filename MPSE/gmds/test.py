import gd, mds, mdsq, mmdsq, special_test, assignment

#random_disk(feedback=True)
#random_ssphere(feedback=True)

### gd ###

#gd.example1()
#gd.example2()
#gd.example3()

### mds ###

#mds.example_simple()
#mds.example_rate(n=10,p=3,min_rate=-8,max_rate=-2)
#mds.example_rate(100,3,-12,-4)
#mds.example_rate(1000,3,-12,-6)
#mds.example_initial(n=10,p=3,rate=0.01,runs=10)
#mds.example_initial(n=100,p=3,rate=0.001,runs=10)

### mdsq ###

#mdsq.example1()

### mmdsq ###

#mmdsq.example_Xdescent()
#mmdsq.example_Qdescent()
mmdsq.example_XQdescent()

########## Test in special_test.py ##########

#mMDSq_Xdescent_Qmultiple_example0()

#mMDSq_Xdescent_noisyQ0()

#special_test.initialization_test0(10,rates=[0.01,0.01])
#special_test.initialization_test0(30,rates=[0.005,0.0005])
#special_test.initialization_test0(100,rates=[0.001,0.0001])
#special_test.initialization_test0(10,randomQs=True,rates=[0.01,0.01])
#special_test.initialization_test0(30,randomQs=True,rates=[0.005,0.0005])
#special_test.initialization_test0(100,randomQs=True,rates=[0.001,0.0001])

#special_test.optimization_test0(n=10,rates=[0.01,0.01])
#special_test.optimization_test0(n=30,rates=[0.005,0.0005])
#special_test.optimization_test0(n=100,rates=[0.001,0.0001])

#assignment.test0()
