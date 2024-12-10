const videoElement = document.getElementById('webcam');
const labelsContainer = document.getElementById('labels');

navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        videoElement.srcObject = stream;
    })
    .catch(err => {
        console.error("Error accessing webcam:", err);
    });

videoElement.onloadedmetadata = () => {
    videoElement.play();
};

function drawLabels(data) {
    labelsContainer.innerHTML = '';
    data.forEach(result => {
        const labelDiv = document.createElement('div');
        labelDiv.className = 'label';
        labelsContainer.appendChild(labelDiv);

        const scaleWidth = videoElement.clientWidth / videoElement.videoWidth;
        const scaleHeight = videoElement.clientHeight / videoElement.videoHeight;

        labelDiv.style.left = (result.face_coords[0] * scaleWidth) + 'px';
        labelDiv.style.top = (result.face_coords[1] * scaleHeight - 20) + 'px';
        labelDiv.textContent = `Emotion: ${result.emotion}, Gender: ${result.gender}, Age: ${result.age}`;
    });
}

function captureAndSendFrame() {
    if (videoElement.readyState >= videoElement.HAVE_ENOUGH_DATA) {
        const canvas = document.createElement('canvas');
        canvas.width = videoElement.videoWidth;
        canvas.height = videoElement.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
        const frameData = canvas.toDataURL('image/jpeg');

        fetch('/process_frame', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ image: frameData })
        })
        .then(response => response.json())
        .then(data => {
            console.log('Received data:', data);
            drawLabels(data);
        })
        .catch(error => {
            console.error('Error sending frame:', error);
        });
    }
}

function startProcessing() {
    function loop() {
        captureAndSendFrame();
        requestAnimationFrame(loop);
    }
    loop();
}

videoElement.onloadedmetadata = () => {
    videoElement.play();
    startProcessing();
};