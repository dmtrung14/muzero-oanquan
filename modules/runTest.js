const { spawn } = require('child_process');

function runTest(scriptPath, args) {
  return new Promise((resolve, reject) => {
    const pyProg = spawn('python', [scriptPath].concat(args));

    let data = '';
    pyProg.stdout.on('data', (stdout) => {
      data += stdout.toString();
    });

    pyProg.stderr.on('data', (stderr) => {
      console.log(`stderr: ${stderr}`);
      reject(new Error(`Python script exited with error: ${stderr}`));
    });

    pyProg.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`Python script exited with code ${code}`));
      } else {
        resolve(data);
      }
    });
  });
}



module.exports = runTest