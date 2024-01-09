const mongoose = require('mongoose')

const userSchema = new mongoose.Schema({
  name: {
    type: String,
    required: [true, 'must provide name'],
    trim: true,
    maxlength: [20, 'name can not be more than 20 characters'],
  },
  email: {
    type: String,
    required: [true, 'must provide email'],
    trim: true,
    maxlength: [50, 'email can not be more than 50 characters'],
  },
  affiliation: {
    type: String,
    trim: true,
    maxlength: [50, 'affiliation can not be more than 50 characters'],
  },
  score: {
    type: Number,
    required: [true, 'must provide score'],
  }
})

const modelSchema = new mongoose.Schema({
    user: {
        type: mongoose.Schema.ObjectId,
        ref: 'User',
        required: true,
    },
    score: {
        type: Number,
        required: [true, 'must provide score'],
    },
    timelog: {
        type: String,
        // required: [true, 'must provide timelog'],
    }
})

const User = mongoose.model('User', userSchema)
const Model = mongoose.model('Model', modelSchema) 

module.exports = { User, Model }