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



const sqlite3 = require('sqlite3').verbose();

// open the database
let db = new sqlite3.Database('mpserecords.sqlite');




app.use('/static', express.static('MPSE/outputs/'))
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

app.post('/run', multipartMiddleware, function (req, res) {
    res.header('Content-Type', 'text/html;charset=utf-8');
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




    sample_size = req.body.sample_size
    preloadeddata = req.body.preloadeddata
    if (req.body.data == "preloaded") {
        var datapath = ""
        if (preloadeddata == 'credit')
            datapath = ['-d', 'MPSE/datasets/dataset_tabluar/data/dissimple1000_1.csv', 'MPSE/datasets/dataset_tabluar/data/dissimple1000_2.csv', 'MPSE/datasets/dataset_tabluar/data/dissimple1000_3.csv']

        if (preloadeddata == '123')
            datapath = ['-d', 'MPSE/datasets/dataset_3D/123_dataset_new/250/data_mat_1_250.csv', 'MPSE/datasets/dataset_3D/123_dataset_new/250/data_mat_2_250.csv', 'MPSE/datasets/dataset_3D/123_dataset_new/250/data_mat_3_250.csv']

        if (preloadeddata == 'circlesquire') {
            req.body.projections = 2;
            //datapath = ['-d', 'MPSE/datasets/dataset_3D/sq_cir_tr_dataset/350/data_mat_cir_350.csv', 'MPSE/datasets/dataset_3D/sq_cir_tr_dataset/350/data_mat_sq_350.csv', 'MPSE/datasets/dataset_3D/sq_cir_tr_dataset/350/data_mat_tr_350.csv']
            datapath = ['-d', 'MPSE/datasets/dataset_3D/circle_square_new/dist_circle.csv', 'MPSE/datasets/dataset_3D/circle_square_new/dist_square.csv']

        }
    }

    if (req.body.data == "uploaded")
        projections_type = "fixed";
    if (req.body.projection_set != "variable")
        projections_type = "fixed";
    else
        projections_type = "variable";


    var parameters = ['mpse.py', '-n', sample_size, '-vt', req.body.vistemplate, '-e', req.body.EXPERIMENT_NAME, '-ps', req.body.projection_set, '-t', projections_type, '-p', req.body.projections, '-max_iters', req.body.max_iters];
    parameters = parameters.concat(datapath)
    console.log("python3.6 " + parameters.join(" "))
    mpse_process = spawn('python3.6', parameters)
    mpse_process.stdout.on('data', function (data) {
        console.log('stdout: ' + data);
        res.write(data + "<br>", 'utf-8');
    });
    mpse_process.stderr.on('data', function (data) {
        console.log('stderr: ' + data);
        db.run("insert into history values('"+req.body.EXPERIMENT_NAME+"')");
        db.close();
        res.write(data, 'utf-8');

    });
    mpse_process.on('exit', function (code) {
        console.log('child process exited with code ' + code);
        res.end(code);
    });
});




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
});
// close the database connection
//db.close();



app.listen(port);
console.log('server is running on', port);
