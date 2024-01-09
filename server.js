const express = require('express');
const runTest = require('./modules/runTest.js');
const connectDB = require('./modules/connect.js');
const fs = require('fs');
const app = express();

const port = process.env.PORT || 3000;


// Render the website using server side rendering
app.use(express.static(__dirname + '/dist'));


app.post('/api/v1/cross_play', (req, res) => {
    // Save the model.checkpoint file
    fs.writeFile('./results/cross_play/model.checkpoint', req.body.file, (err) => {
        if (err) {
            console.error(err);
            res.status(500).send('Error saving file');
        } else {
            res.status(200).send('File saved successfully');
        }
    });

    // Run the test
    runTest('./backend.py', [])
        .then((result) => {
            res.status(200).send(result);
        })
        .catch((err) => {
            res.status(500).send(err);
        });

    // Remove model.checkpoint after performing functions
    fs.unlink('./results/cross_play/model.checkpoint', (err) => {
        if (err) {
            console.error(err);
        } else {
            console.log('model.checkpoint removed successfully');
        }
    });
});

const start = async () => {
    try {
        await connectDB(url);
        app.listen(port, () => {
            console.log('Server started on port ' + port);
        });
    }
    catch (err) {
        console.error(err);
    }
}

start()
