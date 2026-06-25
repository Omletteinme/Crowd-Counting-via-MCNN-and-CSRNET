document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    const uploadPrompt = document.getElementById('upload-prompt');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const removeBtn = document.getElementById('remove-btn');
    const generateBtn = document.getElementById('generate-btn');

    const resultsContent = document.getElementById('results-content');
    const emptyState = document.querySelector('.empty-state');
    const loadingState = document.getElementById('loading-state');
    const successState = document.getElementById('success-state');

    // Result Elements
    const resCount = document.getElementById('res-count');
    const resConfidence = document.getElementById('res-confidence');
    const resTime = document.getElementById('res-time');
    const resDensity = document.getElementById('res-density');

    let currentFile = null;

    // --- File Upload Logic ---

    // Browse Button Click
    browseBtn.addEventListener('click', () => {
        fileInput.click();
    });

    // File Input Change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // Drag and Drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            const file = e.dataTransfer.files[0];
            if (file.type.startsWith('image/')) {
                fileInput.files = e.dataTransfer.files; // assign to input
                handleFile(file);
            } else {
                alert('Please upload an image file (PNG, JPG, JPEG).');
            }
        }
    });

    // Handle File Selection
    function handleFile(file) {
        currentFile = file;
        const reader = new FileReader();
        
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            uploadPrompt.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            generateBtn.disabled = false;
        };
        
        reader.readAsDataURL(file);
    }

    // Remove Image
    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation(); // prevent clicking dropzone
        currentFile = null;
        fileInput.value = '';
        imagePreview.src = '';
        
        previewContainer.classList.add('hidden');
        uploadPrompt.classList.remove('hidden');
        generateBtn.disabled = true;

        // Reset results UI
        resetResults();
    });


    // --- API Request Logic ---
    function resetResults() {
        resultsContent.classList.add('empty');
        emptyState.classList.remove('hidden');
        loadingState.classList.add('hidden');
        successState.classList.add('hidden');
    }

    function showLoading() {
        resultsContent.classList.remove('empty');
        emptyState.classList.add('hidden');
        loadingState.classList.remove('hidden');
        successState.classList.add('hidden');
    }

    function showSuccess(data) {
        loadingState.classList.add('hidden');
        successState.classList.remove('hidden');

        // Animate counter
        animateValue(resCount, 0, data.count, 1000);
        
        resConfidence.textContent = data.confidence;
        resTime.textContent = data.time;
        
        // Display base64 heatmap
        resDensity.src = `data:image/png;base64,${data.density_map}`;
    }

    // Animated counter function
    function animateValue(obj, start, end, duration) {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            obj.innerHTML = Math.floor(progress * (end - start) + start);
            if (progress < 1) {
                window.requestAnimationFrame(step);
            }
        };
        window.requestAnimationFrame(step);
    }

    // Generate Button Click
    generateBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        const modelChoice = document.querySelector('input[name="model"]:checked').value;
        const formData = new FormData();
        formData.append('file', currentFile);
        formData.append('model', modelChoice);

        showLoading();
        generateBtn.disabled = true;

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();
            
            if (data.error) {
                alert(data.error);
                resetResults();
            } else {
                showSuccess(data);
            }
        } catch (error) {
            console.error('Error during prediction:', error);
            alert('An error occurred during estimation. Check console for details.');
            resetResults();
        } finally {
            generateBtn.disabled = false;
        }
    });
});
