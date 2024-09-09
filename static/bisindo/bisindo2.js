const videoInput = document.getElementById("videoInput");
const predictVideoButton = document.getElementById("submit");

predictVideoButton.addEventListener("click", async () => {
  const videoFile = videoInput.files[0];
  if (videoFile) {
    try {
      let formData = new FormData();
      formData.append("video", videoFile);
      // Mengirim video ke server dengan Fetch API
      fetch("/predictBisindo", {
        method: "POST",
        body: formData,
      })
        .then((response) => response.json())
        .then((data) => {
          console.log("Hasil:", data);
          const resultDiv = document.getElementById("predictionResult1");
          resultDiv.textContent = "Huruf yang terdeteksi: " + data["prediction"];
        })
        .catch((error) => {
          console.error("Error:", error);
        });
    } catch (error) {
      console.error("Error loading or predicting with the model:", error);
    }
  } else {
    alert("Please select a video file.");
  }
});

const videoInput1 = document.getElementById("videoInput1");
const predictVideoButton1 = document.getElementById("submit1");

predictVideoButton1.addEventListener("click", async () => {
  const videoFile1 = videoInput1.files[0];
  if (videoFile1) {
    try {
      let formData = new FormData();
      formData.append("video", videoFile1);
      // Mengirim video ke server dengan Fetch API
      fetch("/predictBisindoCLSTM", {
        method: "POST",
        body: formData,
      })
        .then((response) => response.json())
        .then((data) => {
          console.log("Hasil:", data);
          const resultDiv = document.getElementById("predictionResult2");
          resultDiv.textContent = "Huruf yang terdeteksi: " + data["prediction"];
        })
        .catch((error) => {
          console.error("Error:", error);
        });
    } catch (error) {
      console.error("Error loading or predicting with the model:", error);
    }
  } else {
    alert("Please select a video file.");
  }
});
