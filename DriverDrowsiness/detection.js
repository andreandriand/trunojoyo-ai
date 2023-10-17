const videoInput = document.getElementById('videoInput');
const predictVideoButton = document.getElementById('predictVideo');

const SEQUENCE_LENGTH = 20;
const IMAGE_WIDTH = 200;
const IMAGE_HEIGHT=200;

predictVideoButton.addEventListener('click', async () => {
    const videoFile = videoInput.files[0];
    if (videoFile) {
        try {
            // Load the TensorFlow.js model and predict the video.
            const model = await tf.loadLayersModel('DDlstmjsmodel/model.json');
            const data_to_predict = await framesExtraction(videoFile);
            console.log(data_to_predict.shape);
            const predictions = model.predict(data_to_predict);
            console.log(predictions);
        } catch (error) {
            console.error('Error loading or predicting with the model:', error);
        }
    } else {
        alert('Please select a video file.');
    }
});


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
    const videoElement = document.createElement('video');
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
      const canvas = document.createElement('canvas');
      canvas.width = IMAGE_WIDTH;
      canvas.height = IMAGE_HEIGHT;
      const context = canvas.getContext('2d');
      context.drawImage(videoElement, 0, 0, IMAGE_WIDTH, IMAGE_HEIGHT);
      const frameDataUrl = canvas.toDataURL('image/jpeg', 1.0);
  
      // Create an Image element to handle the frame data.
      const imageElement = new Image();
      imageElement.src = frameDataUrl;
  
      // Load the image to ensure it's ready.
      await new Promise((resolve) => (imageElement.onload = resolve));
  
      // Convert the image data to a TensorFlow.js tensor (you may need to adjust this part).
      const tfTensor = tf.browser.fromPixels(imageElement).div(255);
  
      // Append the normalized frame to the frames list.
      framesList.push(tfTensor);
    }
  
    // Remove the video element.
    document.body.removeChild(videoElement);
  
    // Return the frames list.
    return framesList;
}
  
  