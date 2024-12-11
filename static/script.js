
// Access the webcam   Status : OK
navigator.mediaDevices
.getUserMedia({ video: true })
.then(function (stream) {
    document.getElementById("video").srcObject = stream;
})
.catch(function (err) {
    console.log("An error occurred: " + err);
});

// Capture frames and send to the server Status : OK
setInterval(function () {
var video = document.getElementById("video");
var canvas = document.getElementById("canvas");
var context = canvas.getContext("2d");

// Draw the current video frame onto the canvas  Status : OK
context.drawImage(video, 0, 0, 640, 480);

// Convert the canvas content to a base64 image   Status : OK
var dataURL = canvas.toDataURL("image/jpeg", 0.5);

// Send the base64 image to the server   Status : OK
fetch("/process_image", {
    method: "POST",
    headers: {
    "Content-Type": "application/json",
    },
    body: JSON.stringify({ image: dataURL }),
})
    .then((response) => response.json())
    .then((data) => {
    if (data.processed_image) {
        //  Status : OK
        document.getElementById("processedImage").src =
        "data:image/jpeg;base64," + data.processed_image;
        document.getElementById("processedImage").style.display = "block";
    } else if (data.error) {
        console.error("Error:", data.error);
    }
    })
    .catch((error) => console.error("Error:", error));
}, 170); // Status : OK

// All Done 