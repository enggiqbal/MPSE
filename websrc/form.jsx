
import React from 'react';
import axios from 'axios';
import ProgressBar from 'react-bootstrap/ProgressBar'
import { LineChart } from './LineChart'
export class MpseForm extends React.Component {
    constructor(props) {
        super(props);

        this.state = {
            clicked: false,
            resultsLoaded: false,
            error: '',
            hasResults: false,
            response: null,
            data: 'preloaded', average_neighbors: 4,
            sample_size: 200, vistemplate: 'pointbased', EXPERIMENT_NAME: 'exp', projection_type: 'standard', max_iters: 200, smart_initialization: 'random', preloadeddata: 'circlesquare', backgroundcolor: 'black', pointcolor: 'red',

        };
        this.submitQuery = this.submitQuery.bind(this);

        this.onQueryLoad = this.onQueryLoad.bind(this);
        this.myChangeHandler = this.myChangeHandler.bind(this);
    }


    myChangeHandler(event) {

        this.setState({ [event.target.name]: event.target.value })

    }

    submitQuery(event) {

        var bodyFormData = new FormData();

        for (let a in this.state)
            bodyFormData.append(a, this.state[a]);

        this.setState({
            resultsLoaded: false,
            error: '',
            clicked: true
        });

        // axios.post('/run', bodyFormData)
        //     .then(this.onQueryLoad)
        //     .catch(this.onQueryError);

        const config = {

            onDownloadProgress: progressEvent => {

                this.setState({ response: progressEvent.currentTarget.response })
            }

        }
        axios.post('/run', bodyFormData, config).then(this.onQueryLoad);



    }


    onQueryLoad(response) {
        //   console.log(response.data);
        this.setState({
            resultsLoaded: true,
            hasResults: true,
            response: response.data,
        });

    }




    render() {
        return (
            <React.Fragment>
                <form method="post" action="run" encType="multipart/form-data">
                    <div className="container">
                        <Row1 this={this}></Row1>
                        <Row2 this={this}></Row2>
                        <Row3 this={this} ></Row3>
                        <Row4 this={this}></Row4>
                        {this.state.clicked && <Row5 this={this}></Row5>}
                        <Row6 this={this}></Row6>

                        <div className="row">
                            <div className="col-xs-12 col-sm-12 col-md-12 mt-2 mt-sm-2 text-center">
                                <p><u><a href="https://github.com/enggiqbal/MPSE">MPSE github link</a></u></p>
                                
                                <p>Paper: <u><a href="https://arxiv.org/pdf/1909.06485.pdf">Multi-Perspective, Simultaneous Embedding</a></u></p>
                                <p className="h6">© University of Arizona<a className="text-green ml-2" href="http://uamap-dev.arl.arizona.edu:8085/" target="_blank">MPSE web</a></p>
                            </div>
                            <hr />
                        </div>


                    </div>
                </form>
            </React.Fragment>
        );
    }
}

function Row1() {
    return (
        <div className="row">
            <div className="col-md-12  text-center">
                <h1> Multi-Perspective, Simultaneous Embedding (MPSE) Homepage
            </h1>
                <h3> Iqbal Hossain, Vahan Huroyan, Stephen Kobourov, Raymundo Navarrete</h3>
            </div>
        </div>
    );
}

function Row2() {
    return (
        <div className="row top-buffer">
            <div className="col-md-12">
                <div className="card1">
                    <div className="header">
                        Precomputed results
                </div>
                    <div className="list-group">
                        <a href="/precomputed/123" className="list-group-item   ">
                            123 example: (sample size: 10000, max_iters: 500, projection: cylinder average_neighbors: 4,
                            smart initialization: random, visualization template: point based)
                    </a>
                        <a href="/precomputed/circlesquare"
                            className="list-group-item   "> Circling the Square example:
                            (sample size: 200, max_iters: 200, projection: standard, smart
                        initialization: random, visualization template: point based) </a>
                        
                    </div>
                </div>
            </div>
        </div>
    )
}

function Row3(props) {
    let data = props.this.state


    return (
        <div className="row top-buffer">
            <div className="col-md-4  ">
                <div className="card">
                    <div className="header">
                        Data
                    </div>
                    <div className="container">

                        <label className="radio-inline">
                            <input type="radio" name="data" checked={data.data === 'preloaded'} value='preloaded' onChange={props.this.myChangeHandler} ></input>   Use Preloaded Data
                            </label>

                        <select value={data.preloadeddata} name="preloadeddata" onChange={props.this.myChangeHandler} >
                            <option value="123">1-2-3
                                Dataset </option>
                            <option value="credit"  >Credit Dataset
                            </option>
                            <option value="circlesquare">Circle-Square Dataset
                            </option>
                            <option value="cluster">Cluster Dataset
                            </option>
                        </select>




                        <label className="radio-inline"><input type="radio" name="data" value='uploaded' checked={data.data === 'uploaded'} onChange={props.this.myChangeHandler}></input>  Upload
                            Distance
                                Matrices   </label>

                                (<a href="data/dataset_3D/circle_square_new/dist_circle.csv">  Examples of data format</a>) <br />
                        <label> Matrix 1: <input type="file" name="file1"></input></label>
                        <label> Matrix 2: <input type="file" name="file2"></input></label>
                        <label>Matrix 3: <input type="file" name="file3"></input></label>


                    </div>
                </div>
            </div>



            <div className="col-md-4 ">
                <div className="card">
                    <div className="header">
                        Experiment Settings
                    </div>

                    <div className="container">

                        <div className="form-group">
                            <label> Experiment name:</label>
                            <input value={data.EXPERIMENT_NAME} onChange={props.this.myChangeHandler} type="text" name="EXPERIMENT_NAME"></input><br />
                        </div>

                        <div className="form-group">
                            <label> Projections Type:
                            </label> <select defaultValue="standard" name="projection_type" onChange={props.this.myChangeHandler}>
                                <optgroup label="fixed">
                                    <option value="same">same</option>
                                    <option value="standard">standard</option>
                                    <option value="cylinder">cylinder</option>
                                    <option value="orthogonal">orthogonal</option>
                                    <option value="normal">normal</option>
                                    <option value="uniform">uniform</option>
                                </optgroup>
                                <option value="variable">variable</option>
                            </select> </div>
                        <div className="form-group">
                            <label> Max iterations:</label> <input onChange={props.this.myChangeHandler} type="text" value={data.max_iters} name="max_iters"></input>
                        </div>
                        <div className="form-group">
                            <label> Average neighbors: </label><input onChange={props.this.myChangeHandler} type="text" value={data.average_neighbors} name="average_neighbors"></input>
                        </div>
                        <div className="form-group">
                            <label>
                                Sample Data Size:</label> <input onChange={props.this.myChangeHandler} type="text" value={data.sample_size} name="sample_size"></input> </div>
                        <div className="form-group">
                            <label>Smart Initialization: </label><select value={data.smart_initialization} onChange={props.this.myChangeHandler} name="smart_initialization">
                                <option value="mds">mds based</option>
                                <option value="random" >random</option>
                            </select></div>

                    </div>
                </div>
            </div>

            <div className="col-md-4 ">
                <div className="card">
                    <div className="header">
                        3D Vis. Settings
                    </div>

                    <div className="container">
                        <div className="form-group">
                            <label htmlFor="Visualization"> Visualization template:</label>
                            <select name="vistemplate" value={data.vistemplate} onChange={props.this.myChangeHandler}>
                                <option value="pointbased">Point Based</option>
                                <option value="attributebased">Attribute Focused</option>
                            </select>
                        </div>

                        <div className="form-group">
                            <label> Background color: </label>
                            <select name="backgroundcolor" value={data.backgroundcolor} onChange={props.this.myChangeHandler} >
                                <option value="black">Black</option>
                                <option value="white">White</option>
                            </select>
                        </div>
                        <div className="form-group">
                            <label>
                                Point color: </label> <select value={data.pointcolor} name="pointcolor" onChange={props.this.myChangeHandler}>
                                <option value="green">Green</option>
                                <option value="purple">Purple</option>
                                <option value="yellow">Yellow</option>
                            </select>
                        </div>
                    </div>
                </div>
            </div>



        </div>

    )
}

function Row4(props) {


    return (
        <div className="row justify-content-center top-buffer">
            <button type="button" className="btn btn-primary" onClick={props.this.submitQuery}>Run MPSE</button>

        </div>

    )
}


function Row5(props) {


    let response = props.this.state.response;
    //console.log(response);
    let finalStressInfo=""

    //const regex = /<br>\s+(\d+).(\d+).:.cost.=(.*?),/gm;
    const regex = /(\d+).\s(\d+)\s+(.*?)\s/gm
    let m;
    let steps = 0
    let totalstep = 0;
    let data = []
    
    while ((m = regex.exec(response)) !== null) {
        if (m.index === regex.lastIndex) {
            regex.lastIndex++;
        }
        steps = parseInt(m[1]);
        totalstep = parseInt(m[2]);
 
        data.push({ a: steps, b: parseFloat(m[3]) })
    }
    
    let expname = null;
 
    if (props.this.state.resultsLoaded && response.includes('cost.png'))
        {expname = props.this.state.EXPERIMENT_NAME;
            console.log(response)
        finalStressInfo=response.substr( response.indexOf("<b>Final Stress</b>"), response.indexOf("##")-response.indexOf("<b>Final Stress</b>"))
        finalStressInfo=finalStressInfo.replace("##","")
    }

    const width = 600, height = 350, margin = 20



    return (
        [
            <div className="row justify-content-center top-buffer">
                <div className="col-md-12" height="200px">
                    {totalstep ? <ProgressBar now={steps + 1} max={totalstep} label={'steps ' + `${steps }` + ' of ' + totalstep} /> : "please wait"}
                </div>
            </div>,
            <div className="row justify-content-center top-buffer">
                <div className="col-md-6"  >
                    <div className="card">
                        <div className="header">
                            Cost History
                    </div>
                        <div className="container">
                            <LineChart data={data} width={width} height={height} />
                        </div>
                    </div>
                </div>
                <div className="col-md-6">
                    <OutputLinks expname={expname} finalStressInfo={finalStressInfo}></OutputLinks>
                </div>
            </div>
        ]
    )
}

function OutputLinks(props) {
    let expname = props.expname;
    let finalStressInfo=props.finalStressInfo
 
    return (
        <div className="card">
            <div className="header">
                Outputs
        </div>
            <div className="container">
                {expname &&
                    <div>
                        <a target='_blank' href={'static/' + expname + '/index.html'}>Interactive visualization</a>
                        <br></br>
                        <a target='_blank' href={'static/' + expname + '/' + expname + '_pos.csv'}>Download 3D positions</a>
                        <br></br>
                        <div dangerouslySetInnerHTML={{__html: finalStressInfo}} />
                        
                    </div>
                }
            </div>
        </div>
    )
}


function Row6() {

    return (<div className="row top-buffer">
        <div className="col-md-12">
            <div className="card1">
                <div className="header">
                    Video
        </div>
                <div className="container">


                    <div className="embed-responsive embed-responsive-16by9">
                        <iframe className="embed-responsive-item" src="https://www.youtube.com/embed/KJEjlhFy_p4"></iframe>
                    </div>

                </div>
            </div>
        </div>
    </div>)
}