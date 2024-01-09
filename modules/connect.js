const mongoose = require('mongoose')

const connectDB = (url) => {
  return mongoose.connect(url).then(() => {
    console.log('MongoDB Connected...')
  }).catch((err) => {
    console.error(err)
  })
}

module.exports = connectDB