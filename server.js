var express = require('express');
var app = express();
var spawn = require('child_process').spawn;
var port=8085 ;
// set the view engine to ejs
app.set('view engine', 'ejs');
const bodyParser = require("body-parser");
app.use(bodyParser.urlencoded({
    extended: true
}));
//app.use(express.static('MPSE/outputs/'))
app.use('/static', express.static('MPSE/outputs/'))
// use res.render to load up an ejs view file

// index page 
app.get('/', function(req, res) {
    res.render('home');
});

// about page 
app.get('/test', function(req, res) {




    const { exec } = require('child_process');
exec('python3 test.py', (err, stdout, stderr) => {
  if (err) {
    // node couldn't execute the command
    return;
  }
  // the *entire* stdout and stderr (buffered)
  console.log(`stdout: ${stdout}`);
  console.log(`stderr: ${stderr}`);
  res.send(stdout)
});


    //res.send(stdout)
});



app.post('/run', function(req, res){  
    res.header('Content-Type','text/html;charset=utf-8');
    //
    //python3 mpse.py -d MPSE/datasets/dataset_tabluar/data/dissimple1000_1.csv  MPSE/datasets/dataset_tabluar/data/dissimple1000_2.csv  MPSE/datasets/dataset_tabluar/data/dissimple1000_3.csv -n 10 -max_iters 20
    console.log(req.body)
    sample_size=req.body.sample_size
    mpse_process = spawn('python3', ['mpse.py', '-d', 'MPSE/datasets/dataset_tabluar/data/dissimple1000_1.csv', '-n', sample_size])// -d MPSE/datasets/dataset_tabluar/data/dissimple1000_1.csv  MPSE/datasets/dataset_tabluar/data/dissimple1000_2.csv  MPSE/datasets/dataset_tabluar/data/dissimple1000_3.csv -n 10 -max_iters 20']);
    mpse_process.stdout.on('data', function(data) {
        console.log('stdout: ' + data);
        res.write(data + "<br>" , 'utf-8');
    });
    mpse_process.stderr.on('data', function(data) {
        console.log('stderr: ' + data);
        res.write(data, 'utf-8');
    });
    mpse_process.on('exit', function(code) {
        console.log('child process exited with code ' + code);
        res.end(code);
    });
  });


 

app.listen( port);
console.log('server is running on', port);