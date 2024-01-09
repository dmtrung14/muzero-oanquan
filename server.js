const express = require('express');
const runTest = require('./modules/runTest.js');
const connectDB = require('./modules/connect.js');
const fs = require('fs');
const app = express();
const { User, Model } = require('./modules/schemas');
const multer = require('multer')

const port = process.env.PORT || 3000;


// Render the website using server side rendering
app.use(express.static(__dirname + '/docs'));
app.use(express.json());

const storage = multer.memoryStorage();
const upload = multer({ storage: storage, dest: './results/cross_play/' });



app.post('/api/v1/cross_play', upload.single("modelFile"), (req, res) => {
    // Run the test
    runTest('./backend.py', [])
        .then((result) => {
            User.create({
                name: req.body.name,
                email: req.body.email,
                affiliation: req.body.affiliation,
                score: result,
            })
            res.status(200).send(result);
        })
        .catch((err) => {
            res.status(500).send(err);
        });
    
});

const start = async () => {
    try {
        url = "mongodb+srv://dmtrung14:dangminhxu@cluster0.stal6zb.mongodb.net/muzero-oanquan?retryWrites=true&w=majority"
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
