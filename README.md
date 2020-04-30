# MPSE

1. Install NodeJS 10.16.0. (download from nodejs website)
2. Run `npm install` in this directory to install the dependencies.
3. Run `npm run build` to build the static files.
4. Run `nodemon server.js ` to start the server


# Requirements 
``` console
python3.6 
```

# Jupyter Notebook examples 

MPSE/mview_examples

# Without nodejs server 
``` console
 usage: mpse.py [-h] -d D [D ...] [-o OUTPUT_DIR] [-e EXPERIMENT_NAME]
               [-max_iters MAX_ITERS] [-n SAMPLE_SIZE] [-X0 {True,False}]
               [-ps {fixed,same,standard,cylinder,orthogonal,normal,uniform,variable}]
               [-vt {pointbased,attributebased}] [-an AVERAGE_NEIGHBORS]
               [-ds PRELADED_DATASET]

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
  -ds PRELADED_DATASET, --preladed_dataset PRELADED_DATASET
                        Preladed Dataset
                        
                        
```