

var express = require('express');
var app = express();
var spawn = require('child_process').spawn;
var port = 8085;
var fs = require('fs');
app.set('view engine', 'ejs');
const bodyParser = require("body-parser");
app.use(bodyParser.urlencoded({
    extended: true
}));

var multipart = require('connect-multiparty');
var multipartMiddleware = multipart();



//const sqlite3 = require('sqlite3').verbose();

// open the database
//let db = new sqlite3.Database('mpserecords.sqlite');



app.use('/websrc', express.static('websrc/'))
app.use('/static', express.static('MPSE/outputs/'))
app.use('/precomputed', express.static('MPSE/precomputed/'))

app.use('/data', express.static('MPSE/datasets/'))
app.use('/MPSE', express.static('MPSE/MPSE/'))
app.get('/', function (req, res) {
    res.render('home');
});

function uploader(tmp_path, target_path, res) {
    fs.rename(tmp_path, target_path, function (err) {
        if (err) throw err;
        fs.unlink(tmp_path, function () {
            if (err) throw err;
            //res.write('File uploaded to: ' + target_path);
        });
    });

}
let parserResponse = (txt) => {

    let data = [];
    const regex = /(\d+).(\d+)\s:\scost\s=\s(.*?),\sgrad\s=\s(.*?),\slr\s=\s(.*?),\sstep\s=/gm;
    let m;
    while ((m = regex.exec(txt)) !== null) {
        if (m.index === regex.lastIndex) {
            regex.lastIndex++;
        }
        for (let i = 0; i < 5; i++) {
            data.push(parseFloat( m[i + 1]));
        }
    }
   // console.log(data)
    return data
}
app.post('/run', multipartMiddleware, function (req, res) {

    res.header('Content-Type', 'text/html;charset=utf-8');
    let datapath = [];
    if (req.body.data == "uploaded") {
        var f1 = './uploaded/dist1.csv'
        var f2 = './uploaded/dist2.csv'
        var f3 = './uploaded/dist3.csv'
        datapath = ['-d']
        if (req.files.file1.size > 0) {
            uploader(req.files.file1.path, f1, res);
            datapath.push(f1)
        }
        if (req.files.file2.size > 0) {
            uploader(req.files.file2.path, f2, res);
            datapath.push(f2)
        }
        if (req.files.file3.size > 0) {
            uploader(req.files.file3.path, f3, res);
            datapath.push(f3)
        }

    }




    sample_size = req.body.sample_size;
    preloadeddata = req.body.preloadeddata;
    projection_type = req.body.projection_type;

    if (req.body.data == "preloaded") {
    }

    var parameters = ['mpse.py', '-n', sample_size, '-vt', req.body.vistemplate, '-e', req.body.EXPERIMENT_NAME, '-ps', projection_type, '-max_iters', req.body.max_iters, '-X0', req.body.smart_initialization, '-ds', preloadeddata, '-bgcolor', req.body.backgroundcolor, '-pcolor', req.body.pointcolor];
    console.log(parameters);
    parameters = parameters.concat(datapath)
    console.log("python3.6 " + parameters.join(" "))
    mpse_process = spawn('python3.6', parameters)
    mpse_process.stdout.on('data', function (data) {
       // console.log('stdout: ' + data);
        var fs = require('fs');
      
        var proj =  fs.readFileSync('MPSE/outputs/' + req.body.EXPERIMENT_NAME + "/temp_proj.json", 'utf8');
        var pos =  fs.readFileSync('MPSE/outputs/' + req.body.EXPERIMENT_NAME + "/temp_pos.json", 'utf8');

        let stepDetails = parserResponse("" + data)
    
        let string=JSON.stringify({ stepDetails: stepDetails, proj: proj, pos: pos });
        console.log(string);
        res.write(string);
        //res.flush();

    });
    mpse_process.stderr.on('data', function (data) {
        console.log('stderr: ' + data);


        res.write(data, 'utf-8');

    });
    mpse_process.on('exit', function (code) {
        console.log('child process exited with code ' + code);
        res.end("" + code);
    });

});


/* 

app.get('/history', function (req, res) {

    let sql = `SELECT  expname FROM history`;

    db.all(sql, [], (err, rows) => {
        if (err) {
            throw err;
        }
        rows.forEach((row) => {
            console.log(row.name);
        });
    });
    res.send("list")
}); */
// close the database connection
//db.close();



app.listen(port);
console.log('server is running on', port);
