
function fetchSampleData() {
    fetch('/api/sample_endpoint')
        .then(response => response.json())
        .then(data => {
            document.getElementById('apiResponse').innerText = data.message;
        });
}
