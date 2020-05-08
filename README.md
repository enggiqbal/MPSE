# MPSE: with webserver (optional)

1. Install NodeJS 10.16.0. (download from nodejs website)
2. Run `npm install` in this directory to install the dependencies.
3. Run `npm run build` to build the static files.
4. Run `nodemon server.js` to start the server

## list of dependencies at python3.6

```
matplotlib==2.2.2
networkx==2.1
pandas==0.23.3
numpy==1.14.5
scipy==1.1.0
autograd==1.2
torch==1.0.1.post2
scikit_learn==0.21.3

```

```console
pip3.6 install -r requirements.txt
```

# MPSE: Jupyter Notebook examples

```console
MPSE/mview_examples
```

# MPSE: python command line

```console

usage: mpse.py [-h] -d D [D ...] [-o OUTPUT_DIR] [-e EXPERIMENT_NAME]
               [-max_iters MAX_ITERS] [-n SAMPLE_SIZE] [-X0 {True,False}]
               [-ps {fixed,same,standard,cylinder,orthogonal,normal,uniform,variable}]
               [-vt {pointbased,attributebased}] [-an AVERAGE_NEIGHBORS]
               [-ds PRELOADED_DATASET]

MPSE

optional arguments:
  -h, --help            show this help message and exit
  -d D [D ...], --d D [D ...]
                        List of input files with distace matices
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        Output directory
  -e EXPERIMENT_NAME, --experiment_name EXPERIMENT_NAME
                        Experiment name
  -max_iters MAX_ITERS, --max_iters MAX_ITERS
                        Max iterations
  -n SAMPLE_SIZE, --sample_size SAMPLE_SIZE
                        Number of samples
  -X0 {True,False}, --X0 {True,False}
                        Smart initialization
  -ps {fixed,same,standard,cylinder,orthogonal,normal,uniform,variable}, --projection_type {fixed,same,standard,cylinder,orthogonal,normal,uniform,variable}
                        projection set
  -vt {pointbased,attributebased}, --visualization_template {pointbased,attributebased}
                        Visualization template
  -an AVERAGE_NEIGHBORS, --average_neighbors AVERAGE_NEIGHBORS
                        average neighbors
  -ds PRELOADED_DATASET, --preloaded_dataset PRELOADED_DATASET
                        Preloaded Dataset


```

## examples of command

pass distance matrices

```console
python3 mpse.py -d MPSE/datasets/dataset_tabluar/data/dissimple1000_1.csv  MPSE/datasets/dataset_tabluar/data/dissimple1000_2.csv  MPSE/datasets/dataset_tabluar/data/dissimple1000_3.csv
```

run circlesquare example with 150 points, maximum iteration 100 then save output to mytest directory.

```console
python3.6 mpse.py -ds circlesquare -n 150 -max_iters 100 -e mytest

iqbal@on-campus-10-138-77-23 MPSE-web % python3.6 mpse.py -ds circlesquare -n 150 -max_iters 100 -e mytest
<h1>Please keep the window running</h1>
Total Samples found:100<br>
mpse.MPSE():
multigraph.DISS():
  nodes : 100
  added attribute:
    type : matrix
    complete : True
  added attribute:
    type : matrix
    complete : True
  MPSE.initialize():
    X0 : random
    Q0 : given
  dissimilarity stats:
    number of views : 2
    number of points : 100
  embedding stats:
    embedding dimension : 3
    projection dimension : 2
  MPSE.gd():
    mpse method : fixed projections
    initial stress : 5.89e-01
gd.single():
  computation parameters:
    stochastic : True
    constraint : False
    scheme : mm
    initial lr : 1
    min_cost : 1.00e-03
    max_iter : 100
    max_step : 1.00e+10
  progress:
      99/100 : cost = 4.31e-02, grad = 4.97e-03, lr = 2.20e+00, step = 1.09e-02
  results:
    conclusion : maximum number of iterations reached
    total iterations : 99
    final cost : 4.31e-02
    final gradient size : 4.97e-03
    final learning rate : 2.20e+00
    final step size : 1.09e-02
    time : 1.37e+01 [sec]
  Final stress : 4.27e-02
Saving js data in: MPSE/outputs/mytest/coordinates.js
JS file was saved in:  MPSE/outputs/mytest/coordinates.js
DEPRECATION WARNING: The system version of Tk is deprecated and may be removed in a future release. Please don't rely on it. Set TK_SILENCE_DEPRECATION=1 to suppress this warning.
**<br> output path: MPSE/outputs/mytest/index.html**
<br><h1> <a target='_blank'  href ='static/mytest/index.html'>Interactive visualization</a></h1><br>
<br><h2> <a target='_blank'  href ='static/mytest/mytest_pos.csv'>Output 3D position was saved here</a></h2><br>
<br><h2> <a target='_blank'  href ='static/mytest/coordinates.js'>Output details (history, projections, position) was saved here</a></h2><br>
<br>cost history saved as  MPSE/outputs/mytest/cost.png
<br><img src=/static/mytest/cost.png>
```

```
python3.6 mpse.py -ds 123 -max_iters 500 -ps cylinder -e 123 -an 4
```

# Citation (draft version )

```console
@misc{hossain2019multiperspective,
    title={Multi-Perspective, Simultaneous Embedding},
    author={Md Iqbal Hossain and Vahan Huroyan and Stephen Kobourov and Raymundo Navarrete},
    year={2019},
    eprint={1909.06485},
    archivePrefix={arXiv},
    primaryClass={cs.DS}
}
```
