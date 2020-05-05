import ReactDOM from 'react-dom';
import React from 'react';

export class MpseForm extends React.Component {
    constructor(props) {
        super(props);
this.state={

};
    }

render(){
    return(
<form method="post" action="run" enctype="multipart/form-data">
    <div class="container">
        <div class="row">
            <div class="col-md-12">
                <h1> Multi-Perspective, Simultaneous Embedding (MPSE) Homepage
                </h1>
            </div>
        </div>
        <div class="row top-buffer">
            <div class="col-md-12">
                <div class="card1">
                    <div class="header">
                        Precomputed results
                    </div>
                    <div class="list-group">
                        <a href="/precomputed/123" class="list-group-item list-group-item-action list-group-item-light">
                            123 example (sample size: 10000, max_iters: 500, projection: cylinder average_neighbors: 4,
                            smart initialization: random, visualization template: point based)
                        </a>
                        <a href="/precomputed/circlesquire"
                            class="list-group-item list-group-item-action list-group-item-light"> circlesquire example
                            (sample size: 200, max_iters: 200, projection: standard, smart
                            initialization: random, visualization template: point based) </a>

                        <a href="/MPSE" class="list-group-item list-group-item-action list-group-item-light"> More
                            Examples </a>
                    </div>
                </div>


            </div>
        </div>
        <div class="row top-buffer">

            <div class="col-md-4  ">
                <div class="card">
                    <div class="header">
                        Data
                    </div>

                    <div class="container">

                        <label class="radio-inline">
                        <h3>  <input type="radio" name="data" value="preloaded" checked>
                            Use Preloaded Data</label></h3>
                            <select name="preloadeddata"><option value="123">1-2-3
                                Dataset </option> 
                                <option value="credit">Credit Dataset
                             </option> 
                             <option value="circlesquire">Circle-Squire Dataset
                            </option> 
                            <option value="cluster">Cluster Dataset
                            </option> 
                            </select>
 

                    
                        <h3> <label class="radio-inline"><input type="radio" name="data" value="uploaded"> Upload
                                Distance
                                Matrices</label></h3> (<a href="data/dataset_3D/circle_square_new/dist_circle.csv">
                            Examples of
                            data format</a>) <br>
                            <label > Matrix 1: <input type="file" name="file1"></label>
                            <label > Matrix 2: <input type="file" name="file2"></label>
                                <label >Matrix 3: <input type="file" name="file3"></label>
                    </div>
                </div>
            </div>



            <div class="col-md-4 ">
                <div class="card">
                    <div class="header">
                        Experiment Settings
                    </div>

                    <div class="container">

                        <div class="form-group">
                            <label> Experiment name:</label>
                         <input type="text" value="exp" name="EXPERIMENT_NAME"><br>
                        </div>
                         <!-- number of projections: <input type="text" value="3" name="projections"><br> -->
                         <div class="form-group">
                            <label>  Projections Type:
                            </label> <select name="projection_type">
                            <optgroup label="fixed">
                                <option value="same">same</option>
                                <option selected="selected" value="standard">standard</option>
                                <option value="cylinder">cylinder</option>
                                <option value="orthogonal">orthogonal</option>
                                <option value="normal">normal</option>
                                <option value="uniform">uniform</option>
                            </optgroup>
                            <option value="variable">variable</option>
                        </select>   </div>


                        <!-- learning_rate: <input type="text" value="0.001" name="learning_rate"><br> -->
                          <div class="form-group">
                            <label> Max iterations:</label>  <input type="text" value="100" name="max_iters">
                        </div>
                        <div class="form-group">
                            <label> Average neighbors: </label><input type="text" value="16" name="average_neighbors"></div>
                            <div class="form-group">
                                <label>   
                    Sample Data Size:</label> <input type="text" value="100" name="sample_size"> </div>


                        <div class="form-group">
                            <label>Smart Initialization: </label><select name="smart_initialization">
                            <option value="mds">mds based</option>
                            <option value="random" selected>random</option>
                        </select></div>

                    </div>
                </div>
            </div>

            <div class="col-md-4 ">
                <div class="card">
                    <div class="header">
                        3D Visualization Settings
                    </div>

                    <div class="container">
                        <div class="form-group">
                            <label for="Visualization"> Visualization template:</label>
                            <select name=vistemplate>
                                <option value="pointbased">Point Based</option>
                                <option value="attributebased">Attribute Focused</option>
                            </select>
                        </div>
                     
                        <div class="form-group">
                            <label>  Background color: </label>
                       <select  name="backgroundcolor">
                            <option value="black">Black</option>
                            <option value="white">White</option>
                        </select> 
                    </div>
                    <div class="form-group">
                        <label> 
                        Point color: </label> <select  name="pointcolor">
                            <option value="green">Green</option>
                            <option value="purple">Purple</option>
                            <option value="yellow">Yellow</option>
                        </select>
                    </div>
                    </div>
                </div>
            </div>
            <div>
            </div>

          
        </div>
   

        <div class="row justify-content-center top-buffer">
            <div id="app"></div>
             <script src="/websrc/js/index.bundle.js"></script>
        </div>
    </div>
</form>
);
}
}