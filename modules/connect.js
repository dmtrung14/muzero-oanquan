const mongoose = require('mongoose')

url = "mongodb+srv://dmtrung14:dangminhxu@cluster0.stal6zb.mongodb.net/?retryWrites=true&w=majority"

const connectDB = (url) => {
  return mongoose.connect(url, {
    useNewUrlParser: true,
    useCreateIndex: true,
    useFindAndModify: false,
    useUnifiedTopology: true,
  }).then(() => {
    console.log('MongoDB Connected...')
  }).catch((err) => {
    console.error(err)
  })
}

module.exports = connectDB