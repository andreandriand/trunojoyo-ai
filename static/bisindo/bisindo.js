const videoInput = document.getElementById("videoInput");
const predictVideoButton = document.getElementById("predictVideo");

const SEQUENCE_LENGTH = 50;
const IMAGE_WIDTH = 100;
const IMAGE_HEIGHT = 100;

predictVideoButton.addEventListener("click", async () => {
  const videoFile = videoInput.files[0];
  const classList = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"];
  if (videoFile) {
    try {
      // Load the TensorFlow.js model and predict the video.
      const model = await tf.loadLayersModel("bisindo/model/model.json");
      const data_to_predict = await createDataToPredict(videoFile);
      const predictions = model.predict(data_to_predict);
      // Mengambil nilai dari tensor predictions sebagai array JavaScript
      const predictionsArray = await predictions.array();
      console.log(classList[23]);

      // Menampilkan hasil prediksi, misalnya dalam konsol
      const predictionResult = classList[await getIndexMax(predictionsArray[0])];
      console.log(await getIndexMax(predictionsArray[0]));

      const resultDiv = document.getElementById("predictionResult");
      resultDiv.textContent = "Huruf yang terdeteksi: " + predictionResult;
    } catch (error) {
      console.error("Error loading or predicting with the model:", error);
    }
  } else {
    alert("Please select a video file.");
  }
});

async function getIndexMax(array) {
  let max = array[0];
  let indexMax = 0;
  for (let i = 1; i < array.length; i++) {
    if (array[i] > max) {
      max = array[i];
      indexMax = i;
    }
  }
  return indexMax;
}

async function createDataToPredict(videoFile) {
  // Extract frames from the video file using the framesExtraction function (previously defined).
  const frames = await framesExtraction(videoFile);

  const features = [];

  if (frames.length === SEQUENCE_LENGTH) {
    features.push(frames);
  }

  // Convert features to a TensorFlow.js tensor.
  const tfFeatures = tf.tensor(features);

  return tfFeatures;
}

async function framesExtraction(videoFile) {
  // Create a list to store video frames.
  const framesList = [];

  // Create a video element to load the video file.
  const videoElement = document.createElement("video");
  document.body.appendChild(videoElement);

  // Set the video source to the video file.
  videoElement.src = URL.createObjectURL(videoFile);

  // Wait for the video's metadata to load.
  await new Promise((resolve) => {
    videoElement.onloadedmetadata = resolve;
  });

  // Calculate the interval for frame extraction.
  const videoFramesCount = videoElement.duration * 30; // Assuming 30 frames per second
  const skipFramesWindow = Math.max(Math.floor(videoFramesCount / SEQUENCE_LENGTH), 1);

  // Iterate through the video frames.
  for (let frameCounter = 0; frameCounter < SEQUENCE_LENGTH; frameCounter++) {
    // Set the current time of the video based on the skip frames window.
    videoElement.currentTime = (frameCounter * skipFramesWindow) / 30; // 30 is the assumed frame rate.

    // Capture the frame from the video element.
    const canvas = document.createElement("canvas");
    canvas.width = IMAGE_WIDTH;
    canvas.height = IMAGE_HEIGHT;
    const context = canvas.getContext("2d");
    context.drawImage(videoElement, 0, 0, IMAGE_WIDTH, IMAGE_HEIGHT);
    const frameDataUrl = canvas.toDataURL("image/jpeg", 1.0);

    // Create an Image element to handle the frame data.
    const imageElement = new Image();
    imageElement.src = frameDataUrl;

    // Load the image to ensure it's ready.
    await new Promise((resolve) => (imageElement.onload = resolve));

    // Convert the image data to a TensorFlow.js tensor (you may need to adjust this part).
    const tfTensor = tf.browser.fromPixels(imageElement).div(255);

    const arraybiasa = tfTensor.arraySync();

    // Append the normalized frame to the frames list.
    framesList.push(arraybiasa);
  }

  // Remove the video element.
  document.body.removeChild(videoElement);

  // Return the frames list.
  return framesList;
}
