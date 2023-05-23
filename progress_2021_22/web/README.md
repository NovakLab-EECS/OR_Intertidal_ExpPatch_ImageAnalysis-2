# Web App

The goal of the web app is to better allow future students to classify the images and to view model predictions.

## Installation

There are several steps to get this part of the project working.

### Requirements

- [Node.js](https://nodejs.org/en/)
- [Python](https://www.python.org/)

### Steps

1. ``cd`` to this folder.
  
2. After installing Node.js, run the following command:

```bash
npm install
```

3. After installing Python, run the following command:

```bash
python3 scripts/model-pred-to-json.py
```

4. Run the following command to generate some static files used by the web app:

```bash
node scripts/generateJson.js
```

5. Now there are two possible ways to run the web app.
   1. You can run the web app locally by running the following command: ``npm run dev``.
   2. Or you can build the web app to host on a server by running the following command: ``npm run build``.


## Usage

If running the web app locally, you can access the web app by visiting the following URL [http://localhost:3000/](http://localhost:3000/). The web app will automatically refresh every time you make a change to the code.

If building the web app to host on a server, please upload the ``dist/`` folder to a static serving website.

Once you have navigated to the web app, you can use the following buttons to perform the two actions: ``Verify Inferences`` and ``Draw Bounding Boxes``.

### Verify Inferences

1. Click the ``Verify Inferences`` button.
2. Upload the generated JSON file to the web app.
3. All the predictions will be displayed in the web app.

### Draw Bounding Boxes

1. Click the ``Draw Bounding Boxes`` button.
2. Navigate to the folder by clicking the folder name you wanted.
3. Click on the image you want to draw bounding boxes on.
4. Select a type from the left dropdown menu.
5. Click one corner of the bounding box and then the opposite diagonal corner.
6. Proceed until all the bounding boxes are drawn.
7. Click the ``Save`` button to generate the labeled text file for the model to use in the future.