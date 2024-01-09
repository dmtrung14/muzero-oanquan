const submittedModel = document.getElementById('modelFile');
const submitName = document.getElementById('submit-name');
const submitEmail = document.getElementById('submit-email');
const submitAffiliation = document.getElementById('submit-affiliation');
const submitButton = document.getElementById('submit-model');
const { User, Model} = require("../modules/schemas.js");

submitButton.addEventListener("click", () => {
    const file = submittedModel.files[0];
    // Use FormData to send the file to the server
    const formData = new FormData();
    formData.append("file", file);
    formData.append("name", submitName.value);
    formData.append("email", submitEmail.value);
    formData.append("affiliation", submitAffiliation.value);

    fetch("/api/v1/cross_play", {
        method: "POST",
        body: formData,
    })
    .then(response => response.text())
    .then(result => {
        // upload result to mongoDB
        const user = User.create({
            name: submitName.value,
            email: submitEmail.value,
            affiliation: submitAffiliation.value,
            score: Number(result),
        })

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

