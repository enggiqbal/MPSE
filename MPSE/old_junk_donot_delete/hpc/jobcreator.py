import pandas as pd

list=pd.read_csv( 'joblist.csv', header=None)
f=open("hpc_multiview_job_template.pbs")
pbs=f.read()

runtext=""

for i in range(0, list.shape[0]):
    job_name=list[0][i].strip()
    data_path=list[1][i].strip()
    print(data_path)
    pbs_job=open("pbs/"+job_name+".pbs", "w")
    pbs_tmp=pbs.replace("@@@job_name@@@",job_name )
    pbs_tmp=pbs_tmp.replace("@@@data_path@@@",data_path )
    pbs_tmp=pbs_tmp.replace("@@@projection_set@@@",job_name[::-1][0] )
    pbs_tmp=pbs_tmp.replace("@@@number_of_weights@@@",str(len(job_name.split("_")[1].split("p")[0])) )
    pbs_job.write(pbs_tmp)
    pbs_job.close()
    runtext=runtext+ " qsub pbs/" + job_name+".pbs \n"

r=open("hpc_jobs_run.txt", "w")
r.write(runtext)
r.close()
