const path = require('path');
 

     
module.exports = {
    mode: 'development',
    entry: {
        'index': './websrc/index.jsx' 
    },
    output: {
        path: path.resolve('./websrc/js'),
        filename: '[name].bundle.js'
    },
    module: {
        rules: [
            {
                test: /\.jsx?$/,
               loader: 'babel-loader',
                exclude: /node_modules/,
               
                query:
                  {
                    presets:['react']
                  }

            },
            {
                test: /\.geojson$/,
                loader: 'json-loader'
            }
        ]
    },
    resolve: {
        extensions: ['.js', '.jsx']
    }
};