let videoStream;

function startLiveDetection() {
    document.getElementById('live-detection').style.display = 'block';
    document.getElementById('custom-detection').style.display = 'none';

    // Access webcam
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            videoStream = stream;
            const video = document.getElementById('video');
            video.srcObject = stream;

            // Start live detection logic
            // Here you can call your live.java functionality
        })
        .catch(error => {
            console.error('Error accessing webcam:', error);
        });
}

function stopLiveDetection() {
    if (videoStream) {
        const tracks = videoStream.getTracks();
        tracks.forEach(track => track.stop());
        videoStream = null;
        document.getElementById('live-detection').style.display = 'none';
    }
}

function startCustomDetection() {
    document.getElementById('live-detection').style.display = 'none';
    document.getElementById('custom-detection').style.display = 'block';
}

function handleFileUpload(event) {
    const file = event.target.files[0];
    const reader = new FileReader();
    reader.onload = function(e) {
        const img = document.getElementById('uploadedImage');
        img.src = e.target.result;
        img.style.display = 'block';
    };
    reader.readAsDataURL(file);
}

function runCustomDetection() {
    const input = document.getElementById('imageUpload');
    if (input.files && input.files[0]) {
        const formData = new FormData();
        formData.append('file', input.files[0]);

        // Send file to server for detection
        // Replace '/detect' with the endpoint where your custom.java logic is hosted
        axios.post('/detect', formData)
            .then(response => {
                console.log('Detection result:', response.data);
                // Display detection results on the image
            })
            .catch(error => {
                console.error('Error running custom detection:', error);
            });
    }
}
