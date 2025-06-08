document.addEventListener("DOMContentLoaded", () => {
  const imageUpload = document.getElementById("imageUpload");
  const imagePreview = document.getElementById("imagePreview");
  const predictButton = document.getElementById("predictButton");
  const predictionResult = document.getElementById("predictionResult");

  let selectedFile = null;

  imageUpload.addEventListener("change", (event) => {
    selectedFile = event.target.files[0];
    if (selectedFile) {
      const reader = new FileReader();
      reader.onload = (e) => {
        imagePreview.src = e.target.result;
        imagePreview.style.display = "block";
      };
      reader.readAsDataURL(selectedFile);
      predictionResult.innerHTML = "<p>Prediction will appear here</p>";
    } else {
      imagePreview.style.display = "none";
      imagePreview.src = "#";
      selectedFile = null;
    }
  });

  predictButton.addEventListener("click", async () => {
    if (!selectedFile) {
      predictionResult.innerHTML =
        '<p class="error">Please select an image first!</p>';
      return;
    }

    predictionResult.innerHTML = "<p>Predicting...</p>";

    const formData = new FormData();
    formData.append("image", selectedFile);

    try {
      const response = await fetch("http://localhost:5000/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        let errorMsg = `API Error: ${response.status} ${response.statusText}`;
        try {
          const errorData = await response.json();
          if (errorData && errorData.error) {
            errorMsg = `API Error: ${errorData.error}`;
          }
        } catch (e) {}
        throw new Error(errorMsg);
      }

      const data = await response.json();

      if (data.prediction && data.confidence) {
        const confidenceValue = parseFloat(data.confidence) * 100;
        predictionResult.innerHTML = `
                    <p><strong>Prediction:</strong> ${data.prediction}</p>
                    <p><strong>Confidence:</strong> ${confidenceValue.toFixed(
                      2
                    )}%</p>
                `;
      } else if (data.error) {
        predictionResult.innerHTML = `<p class="error">Error: ${data.error}</p>`;
      } else {
        predictionResult.innerHTML =
          '<p class="error">Unexpected response from API.</p>';
      }
    } catch (error) {
      console.error("Error during prediction:", error);
      predictionResult.innerHTML = `<p class="error">Error: ${error.message}</p>`;
    }
  });
});
