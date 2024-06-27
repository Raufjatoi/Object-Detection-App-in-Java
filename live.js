// Establish WebSocket connection to Java backend
const socket = new WebSocket('ws://localhost:8080/liveDetection');

socket.onopen = function(event) {
    console.log('WebSocket connection established.');
};

socket.onerror = function(error) {
    console.error('WebSocket error:', error);
};

socket.onmessage = function(event) {
    const data = JSON.parse(event.data);
    updateDetectionResults(data.object);
};

function updateDetectionResults(object) {
    const resultsPlaceholder = document.getElementById('results-placeholder');
    resultsPlaceholder.innerHTML = `<strong>Detected Object:</strong> ${object}`;
}
