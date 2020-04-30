# MPSE


1. Install NodeJS 10.16.0. (download from nodejs website)
2. Run `npm install` in this directory to install the dependencies.
3. Run `npm run build` to build the static files.
4. Run `nodemon server.js ` to start the server

# without nodejs server 
``` console
 python3 mpse.py --d MPSE/datasets/dataset_3D/circle_square_new/dist_circle.csv MPSE/datasets/dataset_3D/circle_square_new/dist_square.csv --experiment_name 'exp' --lr 0.0001  --max_iters 100 --output_dir outputs --projection_set 'standard' --projections 2 --projections_type fixed --sample_size 100 --save_progress 0 --verbose 2 --visualization_template 'pointbased'

```