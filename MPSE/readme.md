# installation

## list of dependency 
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
pip3 install -r requirements.txt
```

# uses 
```console

usage: mpse.py [-h] -d D [D ...] [-o OUTPUT_DIR] [-e EXPERIMENT_NAME]
               [-p {1,2,3}] [-t {fixed,variable}] [-ps PROJECTION_SET]
               [-lr LEARNING_RATE] [-max_iters MAX_ITERATIONS]
               [-n SAMPLE_SIZE] [-X0 INITIALIZATION] [-sp SAVE_PROGRESS]
               [-v VERBOSE] [-alg {classic,gd,gdm,agd}]

MPSE

optional arguments:
  -h, --help            show this help message and exit
  -d D [D ...], --d D [D ...]
                        List of input files with distace matices
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        Output directory
  -e EXPERIMENT_NAME, --experiment_name EXPERIMENT_NAME
                        Experiment name
  -p PROJECTIONS, --projections PROJECTIONS
                        Number of projections to optimize
  -t {fixed,variable}, --projections_type {fixed,variable}
                        Projection type
  -ps PROJECTION_SET, --projection_set PROJECTION_SET
                        file for projection set for fixed projection, see
                        examples in resource directory
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        Learning rate
  -max_iters MAX_ITERATIONS, --max_iterations MAX_ITERATIONS
                        Max iterations
  -n SAMPLE_SIZE, --sample_size SAMPLE_SIZE
                        Number of samples
  -X0 INITIALIZATION, --initialization INITIALIZATION
                        Initial initialization, a csv file with 3D co-
                        ordinates
  -sp SAVE_PROGRESS, --save_progress SAVE_PROGRESS
                        save progress
  -v VERBOSE, --verbose VERBOSE
                        verbose
  -alg {classic,gd,gdm,agd}, --algorithm {classic,gd,gdm,agd}
                        algorithms: 'classic' for autograd implementation,
                        'gd' for gradient descent, 'gdm' for GD with momentum,
                        'agd' for adaptive GD

```
## example 1: generic parameters
``` console
python3 mpse.py -d d1.csv d2.csv d3.csv -o testout -e face -p 3 -lr 0.001 -n 1000 -X0 init_co-ordinate.csv -t variable
```
## example 2: example with projection set
```console
python3.6 mpse.py -d MPSE/datasets/dataset_tabluar/data/dissimple1000_1.csv  MPSE/datasets/dataset_tabluar/data/dissimple1000_2.csv  MPSE/datasets/dataset_tabluar/data/dissimple1000_3.csv -n 10 -max_iters 20 -ps resources/fixed_projection_2.txt 
```

# output
 The code `mpse.py` generates outputs 3D co-ordinates, cost histories, javascript files. Default location of the output is `outputs`. If experiment name is provided then output files include `EXPERIMENT_NAME` as prefix in the file name.