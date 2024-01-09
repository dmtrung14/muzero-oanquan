const submittedModel = document.getElementById('modelFile');
const submitName = document.getElementById('submit-name');
const submitEmail = document.getElementById('submit-email');
const submitAffiliation = document.getElementById('submit-affiliation');
const submitButton = document.getElementById('submit-model');
// const { User, Model } = require('../modules/schemas');

console.log(submitButton)

submitButton.addEventListener("click", () => {
    console.log("button clicked")
    const file = submittedModel.files[0];
    // Use FormData to send the file to the server
    const formData = {
        file: file,
        name: submitName.value,
        email: submitEmail.value,
        affiliation: submitAffiliation.value,
    }
 
    console.log(formData)

    // return;

    fetch("/api/v1/cross_play", {
        method: "POST",
        body: JSON.stringify(formData),
        headers: {
            'Content-Type': 'application/json',
        }
    })
    .then(response =>{
        // upload result to mongoDB
        const leaderboard = document.getElementById('leaderboard');
        leaderboard.innerHTML = '';
        const users = User.find().sort({ score: -1 }).limit(50);
        users.forEach((user) => {
            const userEntry = document.createElement('li');
            userEntry.innerHTML = `${user.name} - ${user.score}`;
            leaderboard.appendChild(userEntry);
        })
    })
    .catch(error => {
        console.error("Error:", error);
    });
});

// update leaderboard everytime mongodb database is updated.

