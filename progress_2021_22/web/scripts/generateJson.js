const fs = require('fs');
const path = require('path');
const readline = require('readline').createInterface({
  input: process.stdin,
  output: process.stdout,
});

const IGNORE_FOLDERS = ['1_AnimationStartEndSlides'];
const IMAGE_EXTENSIONS = ['png', 'jpg', 'jpeg', 'gif'];

const obj = {};

readline.question(
  'Enter the path to the folder containing the images: ',
  (folderPath) => {
    const BASE_PATH = folderPath
      ? folderPath
      : '../ExpPatch-Pics/ExpPatchPics-Processed';

    /* 
We want to generate a json file that contains keys that are the folders and then all the images in that folders as an array.
*/

    let filenames = fs.readdirSync(BASE_PATH);
    filenames = filenames.filter(
      (filename) => IGNORE_FOLDERS.indexOf(filename) === -1
    );

    /* 
Traverse through the folders and add them to the json object.
*/

    let totalImages = 0;

    filenames.forEach((filename) => {
      const folderPath = path.join(BASE_PATH, filename);
      let files = fs.readdirSync(folderPath);
      files = files.filter((file) =>
        IMAGE_EXTENSIONS.includes(file.split('.')[1]?.toLocaleLowerCase())
      );
      totalImages += files.length;

      obj[filename] = files;
    });

    console.log(
      `Parsed ${totalImages} images over ${filenames.length} folders`
    );

    // Write the json file
    fs.writeFileSync('./src/public/data.json', JSON.stringify(obj))
    readline.close();
  }
);
