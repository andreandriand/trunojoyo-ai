let video = document.querySelector("video");
let model, model_class, camera;

// declare the canvas variable and setting up the context
let canvas = document.getElementById("canvas");
let ctx = canvas.getContext("2d");

const COLORS = ["green", "blue", "red"];
const LABELS = ["Masker Penuh", "Masker Tidak Penuh", "Tanpa Masker"];

const accessCamera = () => {
  navigator.mediaDevices
    .getUserMedia({
      video: { width: 640, height: 480 },
      audio: false,
    })
    .then((stream) => {
      video.srcObject = stream;
    });
};

const detectFaces = async (faceClass) => {
  const prediction = await model.estimateFaces(video, false);
  ctx.drawImage(video, 0, 0, 640, 480);
  console.log(faceClass);

  prediction.forEach((predictions) => {
    ctx.beginPath();
    ctx.lineWidth = "4";
    ctx.strokeStyle = COLORS[faceClass];
    ctx.rect(predictions.topLeft[0], predictions.topLeft[1], predictions.bottomRight[0] - predictions.topLeft[0], predictions.bottomRight[1] - predictions.topLeft[1]);
    ctx.stroke();

    ctx.font = "15px verdana";
    ctx.fillStyle = COLORS[faceClass];
    ctx.fillText(LABELS[faceClass], predictions.topLeft[0], predictions.topLeft[1] - 15);
  });
};

accessCamera();
video.addEventListener("loadeddata", async () => {
  model = await blazeface.load();
  model_class = await tf.loadLayersModel("tfjs_model/model.json");

  if (camera == undefined) {
    camera = await tf.data.webcam(video);
  }

  setInterval(async () => {
    let image = await camera.capture();
    // let reshaped_image = image.reshape([-1, 32, 32, 3]);
    // let pred = await model_class.predict(reshaped_image);
    // let result = await tf.argMax(pred,axis=1).data();
    // let faceClass = result[0];

    let faceClass = null;
    await tf.tidy(() => {
      let img = image.reshape([-1, 32, 32, 3]);
      img = tf.cast(img, "float32");
      img = img.div(tf.scalar(255));

      const output = model_class.predict(img);
      faceClass = Array.from(output.argMax((axis = 1)).dataSync())[0];
    });
    detectFaces(faceClass);
  }, 40);
});
