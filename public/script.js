const STATUS = document.getElementById('status');
const ENABLE_CAM_BUTTON = document.getElementById('enableCam');
const RESET_BUTTON = document.getElementById('reset');
const TRAIN_BUTTON = document.getElementById('train');
const PREDICTION_RESULTS = document.getElementById('predictionResults');
const MOBILE_NET_INPUT_WIDTH = 224;
const MOBILE_NET_INPUT_HEIGHT = 224;
const STOP_DATA_GATHER = -1;
const CLASS_NAMES = [];

// Webcam elements
const CLASS_WEBCAMS = [
  document.getElementById('webcam1'),
  document.getElementById('webcam2')
];
const PREDICTION_WEBCAM = document.getElementById('predictionWebcam');

ENABLE_CAM_BUTTON.addEventListener('click', enableCam);
TRAIN_BUTTON.addEventListener('click', trainAndPredict);
RESET_BUTTON.addEventListener('click', reset);

function hasGetUserMedia() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

function enableCam() {
  if (hasGetUserMedia()) {
    const constraints = {
      video: true,
      width: 640, 
      height: 480 
    };

    // Activate webcam streams for all class webcams
    navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
      CLASS_WEBCAMS.forEach(webcam => {
        webcam.srcObject = stream.clone();
        webcam.addEventListener('loadeddata', function() {
          videoPlaying = true;
        });
      });
      
      // Activate prediction webcam
      PREDICTION_WEBCAM.srcObject = stream.clone();
      PREDICTION_WEBCAM.addEventListener('loadeddata', function() {
        videoPlaying = true;
      });
      
      ENABLE_CAM_BUTTON.classList.add('removed');
    });
  } else {
    console.warn('getUserMedia() is not supported by your browser');
  }
}

async function trainAndPredict() {
  predict = false;
  tf.util.shuffleCombo(trainingDataInputs, trainingDataOutputs);
  let outputsAsTensor = tf.tensor1d(trainingDataOutputs, 'int32');
  let oneHotOutputs = tf.oneHot(outputsAsTensor, CLASS_NAMES.length);
  let inputsAsTensor = tf.stack(trainingDataInputs);
  
  let results = await model.fit(inputsAsTensor, oneHotOutputs, {
    shuffle: true, 
    batchSize: 5, 
    epochs: 10, 
    callbacks: {onEpochEnd: logProgress} 
  });
  
  outputsAsTensor.dispose();
  oneHotOutputs.dispose();
  inputsAsTensor.dispose();
  predict = true;
  predictLoop();
}

function predictLoop() {
  if (predict) {
    tf.tidy(function() {
      let videoFrameAsTensor = tf.browser.fromPixels(PREDICTION_WEBCAM).div(255);
      let resizedTensorFrame = tf.image.resizeBilinear(videoFrameAsTensor, [
        MOBILE_NET_INPUT_HEIGHT, 
        MOBILE_NET_INPUT_WIDTH
      ], true);

      let imageFeatures = mobilenet.predict(resizedTensorFrame.expandDims());
      let prediction = model.predict(imageFeatures).squeeze();
      let highestIndex = prediction.argMax().arraySync();
      let predictionArray = prediction.arraySync();

      PREDICTION_RESULTS.innerHTML = 'Prediction: <strong>' + CLASS_NAMES[highestIndex] + 
        '</strong> with ' + Math.floor(predictionArray[highestIndex] * 100) + '% confidence';
    });

    window.requestAnimationFrame(predictLoop);
  }
}

function logProgress(epoch, logs) {
  console.log('Data for epoch ' + epoch, logs);
}

function reset() {
  predict = false;
  examplesCount.length = 0;
  for (let i = 0; i < trainingDataInputs.length; i++) {
    trainingDataInputs[i].dispose();
  }
  trainingDataInputs.length = 0;
  trainingDataOutputs.length = 0;
  STATUS.innerText = 'No data collected';
  
  // Reset class progress counters
  document.querySelectorAll('.class-progress').forEach(el => {
    el.textContent = 'Data count: 0';
  });
  
  console.log('Tensors in memory: ' + tf.memory().numTensors);
}

let dataCollectorButtons = document.querySelectorAll('button.dataCollector');
for (let i = 0; i < dataCollectorButtons.length; i++) {
  dataCollectorButtons[i].addEventListener('mousedown', gatherDataForClass);
  dataCollectorButtons[i].addEventListener('mouseup', gatherDataForClass);
  CLASS_NAMES.push(dataCollectorButtons[i].getAttribute('data-name'));
}

let mobilenet = undefined;
let gatherDataState = STOP_DATA_GATHER;
let videoPlaying = false;
let trainingDataInputs = [];
let trainingDataOutputs = [];
let examplesCount = [];
let predict = false;

async function loadMobileNetFeatureModel() {
  const URL = 
    'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1';
  
  mobilenet = await tf.loadGraphModel(URL, {fromTFHub: true});
  STATUS.innerText = 'MobileNet v3 loaded successfully!';
  
  tf.tidy(function () {
    let answer = mobilenet.predict(tf.zeros([1, MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH, 3]));
    console.log(answer.shape);
  });
}

loadMobileNetFeatureModel();

let model = tf.sequential();
model.add(tf.layers.dense({inputShape: [1024], units: 128, activation: 'relu'}));
model.add(tf.layers.dense({units: CLASS_NAMES.length, activation: 'softmax'}));

model.summary();

model.compile({
  optimizer: 'adam',
  loss: (CLASS_NAMES.length === 2) ? 'binaryCrossentropy': 'categoricalCrossentropy', 
  metrics: ['accuracy']  
});

function dataGatherLoop() {
  if (videoPlaying && gatherDataState !== STOP_DATA_GATHER) {
    let activeWebcam = CLASS_WEBCAMS[gatherDataState];
    let imageFeatures = tf.tidy(function() {
      let videoFrameAsTensor = tf.browser.fromPixels(activeWebcam);
      let resizedTensorFrame = tf.image.resizeBilinear(videoFrameAsTensor, [
        MOBILE_NET_INPUT_HEIGHT, 
        MOBILE_NET_INPUT_WIDTH
      ], true);
      let normalizedTensorFrame = resizedTensorFrame.div(255);
      return mobilenet.predict(normalizedTensorFrame.expandDims()).squeeze();
    });

    trainingDataInputs.push(imageFeatures);
    trainingDataOutputs.push(gatherDataState);
    
    if (examplesCount[gatherDataState] === undefined) {
      examplesCount[gatherDataState] = 0;
    }
    examplesCount[gatherDataState]++;
    
    // Update the progress display for this class
    document.querySelectorAll('.class-progress')[gatherDataState].textContent = 
      'Data count: ' + examplesCount[gatherDataState];
    
    window.requestAnimationFrame(dataGatherLoop);
  }
}

function gatherDataForClass() {
  let classNumber = parseInt(this.getAttribute('data-1hot'));
  gatherDataState = (gatherDataState === STOP_DATA_GATHER) ? classNumber : STOP_DATA_GATHER;
  dataGatherLoop();
}