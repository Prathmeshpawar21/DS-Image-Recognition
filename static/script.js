
// In case you want to access webcam directly with JS
window.onload = function() {
    const videoElement = document.createElement('video');
    videoElement.width = 640;
    videoElement.height = 480;
    videoElement.autoplay = true;
    videoElement.controls = true;

    // Access the webcam
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(function(stream) {
            videoElement.srcObject = stream;
            document.body.appendChild(videoElement);
        })
        .catch(function(err) {
            console.log("Error accessing webcam: " + err);
        });
};


