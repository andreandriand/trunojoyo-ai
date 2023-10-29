const videoElement = document.getElementsByClassName('input_video')[0];
const canvasElement = document.getElementsByClassName('output_canvas')[0];
const canvasCtx = canvasElement.getContext('2d');

let model;
const classList = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"];

async function loadModel() {
    model = await tf.loadLayersModel("model-realtime/model.json");
}

loadModel();

async function onResults(results) {
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);
    if (results.multiHandLandmarks) {
        for (const landmarks of results.multiHandLandmarks) {
            drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, { color: '#00FF00', lineWidth: 5 });
            drawLandmarks(canvasCtx, landmarks, { color: '#FF0000', lineWidth: 2 });

            // Lakukan prediksi huruf berdasarkan landmarks
            const predictedLetter = await predictLetter(landmarks);
            console.log('Prediksi Huruf:', predictedLetter);
        }
    }
    canvasCtx.restore();
}

async function predictLetter(landmarks) {
    // Anda perlu mengambil data landmarks tangan dan mengonversinya
    // menjadi bentuk yang sesuai dengan input model Anda.
    
    // Misalnya, jika model Anda memerlukan input berupa array numerik, Anda dapat
    // mengambil landmarks tangan dan mengonversinya menjadi array.
    
    const landmarksArray = [];

    for (const landmark of landmarks) {
        landmarksArray.push(landmark.x, landmark.y, landmark.z);
    }
    // Kemudian, Anda dapat menggunakan model Anda untuk melakukan prediksi
    let inputTensor = tf.tensor([landmarksArray]); // Ubah landmarks menjadi tensor
    inputTensor = inputTensor.expandDims(2); // Tambahkan dimensi batch
    // inputTensor = inputTensor.expandDims(0); // Tambahkan dimensi batch
    // console.log(inputTensor)

    
    // // Gunakan model Anda untuk melakukan prediksi
    const predictions = model.predict(inputTensor);
    // // Ambil hasil prediksi sebagai array JavaScript
    const predictionsArray = await predictions.array();
    
    // Temukan indeks dengan nilai tertinggi sebagai prediksi huruf
    const predictedIndex = await getIndexMax(predictionsArray[0]);
    
    // // Ubah indeks menjadi huruf A-Z
    const predictedLetter = classList[predictedIndex];
    console.log(predictedLetter)

    // return predictedLetter;
}

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

const hands = new Hands({ locateFile: (file) => {
    return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
} });
hands.setOptions({
    maxNumHands: 2,
    modelComplexity: 1,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
});
hands.onResults(onResults);

const camera = new Camera(videoElement, {
    onFrame: async () => {
        await hands.send({ image: videoElement });
    },
    width: 420,
    height: 280
});
camera.start();