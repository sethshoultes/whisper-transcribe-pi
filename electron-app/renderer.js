let isRecording = false;
let mediaRecorder = null;
let audioChunks = [];
let audioContext = null;
let analyser = null;
let microphone = null;
let animationId = null;
let serverReady = false;
let recordingStartTime = null;
const MINIMUM_RECORDING_DURATION = 1000; // 1 second minimum

const recordBtn = document.getElementById('record-btn');
const copyBtn = document.getElementById('copy-btn');
const clearBtn = document.getElementById('clear-btn');
const statusDiv = document.getElementById('status');
const transcriptionArea = document.getElementById('transcription-area');
const loadingDiv = document.getElementById('loading');
const audioVisualizer = document.getElementById('audio-visualizer');
const audioCanvas = document.getElementById('audio-canvas');
const audioLevelSpan = document.getElementById('audio-level');
const progressContainer = document.getElementById('progress-container');
const progressBar = document.getElementById('progress-bar');
const progressText = document.getElementById('progress-text');

// Check if all elements exist
if (!statusDiv) {
    console.error('Status div not found!');
}

const canvasCtx = audioCanvas ? audioCanvas.getContext('2d') : null;

// Get supported mime type for recording
function getSupportedMimeType() {
    const types = [
        'audio/webm;codecs=opus',
        'audio/webm',
        'audio/ogg;codecs=opus',
        'audio/ogg',
        'audio/mp4',
        'audio/mpeg'
    ];
    
    for (const type of types) {
        if (MediaRecorder.isTypeSupported(type)) {
            console.log('Using mime type:', type);
            return type;
        }
    }
    
    console.log('No specific mime type supported, using default');
    return ''; // Let browser choose
}

// Check server health
let serverCheckAttempts = 0;
async function checkServerHealth() {
    serverCheckAttempts++;
    
    try {
        const response = await fetch('http://localhost:5001/health');
        const data = await response.json();
        
        if (data.status === 'ok' && data.model === 'loaded') {
            serverReady = true;
            if (statusDiv) statusDiv.textContent = 'Ready to record';
            if (recordBtn) recordBtn.disabled = false;
            if (progressContainer) progressContainer.style.display = 'none';
            return true;
        } else if (data.status === 'loading') {
            if (statusDiv) statusDiv.textContent = 'Loading Whisper model... (this may take 10-30 seconds)';
            updateProgressBar(serverCheckAttempts, 30);
            return false;
        } else {
            if (statusDiv) statusDiv.textContent = 'Model loading failed';
            if (progressContainer) progressContainer.style.display = 'none';
            return false;
        }
    } catch (error) {
        console.error('Server health check failed:', error);
        
        // During initial startup (first 30 seconds), show loading messages
        if (serverCheckAttempts < 30) {
            updateProgressBar(serverCheckAttempts, 30);
            
            if (statusDiv) {
                if (serverCheckAttempts < 5) {
                    statusDiv.textContent = 'Starting Python server...';
                } else if (serverCheckAttempts < 15) {
                    statusDiv.textContent = 'Loading Whisper model... (this may take 10-30 seconds)';
                } else {
                    statusDiv.textContent = 'Still loading... (large model files)';
                }
            }
        } else {
            // After 30 seconds, assume there's a real problem
            if (statusDiv) statusDiv.textContent = 'Python server not responding. Please restart the app.';
            if (progressContainer) progressContainer.style.display = 'none';
        }
        
        return false;
    }
}

// Update progress bar
function updateProgressBar(current, total) {
    if (progressContainer && progressBar && progressText) {
        progressContainer.style.display = 'block';
        const percentage = Math.min((current / total) * 100, 100);
        progressBar.style.width = percentage + '%';
        progressText.textContent = Math.round(percentage) + '%';
    }
}

// Initialize
async function init() {
    // Show initial loading message
    if (statusDiv) {
        statusDiv.textContent = 'Starting Python server...';
    }
    
    // Start checking server health
    const healthInterval = setInterval(async () => {
        const ready = await checkServerHealth();
        if (ready) {
            clearInterval(healthInterval);
        }
    }, 1000);
    
    // Check if microphone is available
    try {
        // First check if we can access media devices
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error('MediaDevices API not available');
        }
        
        // Request microphone permission
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        // Test that we got a valid stream
        const tracks = stream.getTracks();
        console.log('Got audio tracks:', tracks.length);
        tracks.forEach(track => {
            console.log('Track:', track.kind, track.label, track.readyState);
            track.stop(); // Stop test stream
        });
        
        updateStatus('Ready to record', 'success');
    } catch (error) {
        console.error('Microphone init error:', error);
        updateStatus('Microphone error: ' + error.message, 'error');
        recordBtn.disabled = true;
        
        // Check for specific permission errors
        if (error.name === 'NotAllowedError') {
            updateStatus('Microphone permission denied. Please allow in System Settings.', 'error');
        }
    }
}

// Update status
function updateStatus(message, type = 'info') {
    if (!statusDiv) {
        console.error('Cannot update status - statusDiv is null');
        return;
    }
    statusDiv.textContent = message;
    statusDiv.style.backgroundColor = type === 'error' ? '#ffebee' : 
                                     type === 'success' ? '#e8f5e9' : '#e3f2fd';
    statusDiv.style.color = type === 'error' ? '#c62828' : 
                           type === 'success' ? '#2e7d32' : '#1976d2';
}

// Toggle recording
async function toggleRecording() {
    if (!isRecording) {
        startRecording();
    } else {
        stopRecording();
    }
}

// Start recording
async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
                sampleRate: 16000,  // Standard rate for speech recognition
                channelCount: 1     // Mono audio
            } 
        });
        
        // Get supported mime type
        const mimeType = getSupportedMimeType();
        if (!mimeType) {
            throw new Error('No supported audio format found');
        }
            
        // Configure MediaRecorder with better settings
        const recorderOptions = { 
            mimeType: mimeType,
            audioBitsPerSecond: 128000  // 128kbps for good quality
        };
        
        try {
            mediaRecorder = new MediaRecorder(stream, recorderOptions);
        } catch (e) {
            // Fallback without audioBitsPerSecond if not supported
            console.warn('audioBitsPerSecond not supported, using defaults');
            mediaRecorder = new MediaRecorder(stream, { mimeType: mimeType });
        }
        audioChunks = [];
        
        // Set up audio visualization
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        microphone = audioContext.createMediaStreamSource(stream);
        microphone.connect(analyser);
        analyser.fftSize = 256;
        
        // Show visualizer
        audioVisualizer.style.display = 'block';
        visualizeAudio();
        
        mediaRecorder.ondataavailable = event => {
            if (event.data && event.data.size > 0) {
                audioChunks.push(event.data);
                console.log('Audio chunk received:', event.data.size, 'bytes');
            }
        };
        
        mediaRecorder.onstart = () => {
            console.log('MediaRecorder started');
            console.log('State:', mediaRecorder.state);
            console.log('MimeType:', mediaRecorder.mimeType);
        };

        mediaRecorder.onerror = (event) => {
            console.error('MediaRecorder error:', event.error);
            updateStatus('Recording error: ' + event.error.message, 'error');
        };
        
        mediaRecorder.onstop = async () => {
            // Stop visualization
            cancelAnimationFrame(animationId);
            audioVisualizer.style.display = 'none';
            
            // Clean up audio context
            if (microphone) {
                microphone.disconnect();
                microphone = null;
            }
            if (audioContext) {
                audioContext.close();
                audioContext = null;
            }
            
            // Create audio blob
            const audioBlob = new Blob(audioChunks, { type: mimeType });
            console.log('Created audio blob:', audioBlob.size, 'bytes, type:', mimeType);
            console.log('Total chunks collected:', audioChunks.length);
            
            // Debug: Create audio element to test playback
            const audioUrl = URL.createObjectURL(audioBlob);
            const testAudio = new Audio(audioUrl);
            testAudio.onloadedmetadata = () => {
                console.log('Audio duration:', testAudio.duration, 'seconds');
            };
            
            if (audioBlob.size > 0) {
                await sendAudioToServer(audioBlob);
            } else {
                updateStatus('No audio recorded', 'error');
            }
        };
        
        // Start recording without timeslice to ensure all data is captured
        mediaRecorder.start();
        isRecording = true;
        recordingStartTime = Date.now();
        recordBtn.textContent = 'Stop Recording';
        recordBtn.classList.add('recording');
        updateStatus('Recording... Speak now! (minimum 1 second)', 'info');
        
    } catch (error) {
        updateStatus('Failed to start recording: ' + error.message, 'error');
        console.error('Recording error:', error);
    }
}

// Stop recording
function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        const recordingDuration = Date.now() - recordingStartTime;
        
        // Check minimum recording duration
        if (recordingDuration < MINIMUM_RECORDING_DURATION) {
            updateStatus(`Recording too short (${Math.round(recordingDuration/1000)}s). Please record for at least 1 second.`, 'error');
            
            // Reset recording state
            mediaRecorder.stop();
            mediaRecorder.stream.getTracks().forEach(track => track.stop());
            isRecording = false;
            recordBtn.textContent = 'Start Recording';
            recordBtn.classList.remove('recording');
            
            // Stop visualization
            cancelAnimationFrame(animationId);
            audioVisualizer.style.display = 'none';
            
            // Clean up audio context
            if (microphone) {
                microphone.disconnect();
                microphone = null;
            }
            if (audioContext) {
                audioContext.close();
                audioContext = null;
            }
            
            return;
        }
        
        mediaRecorder.stop();
        mediaRecorder.stream.getTracks().forEach(track => track.stop());
        isRecording = false;
        recordBtn.textContent = 'Start Recording';
        recordBtn.classList.remove('recording');
        updateStatus('Processing...', 'info');
        loadingDiv.style.display = 'block';
    }
}

// Send audio to server
async function sendAudioToServer(audioBlob) {
    try {
        console.log('Sending audio to server...', audioBlob.size, 'bytes');
        
        // Validate audio blob size
        const minSize = 1000; // 1KB minimum
        if (audioBlob.size < minSize) {
            updateStatus(`Audio file too small (${audioBlob.size} bytes). Please try recording for longer.`, 'error');
            return;
        }
        
        // Validate audio blob type
        if (!audioBlob.type || audioBlob.type === '') {
            console.warn('Audio blob has no type specified');
        }
        
        const formData = new FormData();
        formData.append('audio', audioBlob, 'recording.webm');
        
        const response = await fetch('http://localhost:5001/transcribe', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Server error: ${response.status} - ${errorText}`);
        }
        
        const result = await response.json();
        console.log('Server response:', result);
        
        if (result.text) {
            appendTranscription(result.text);
            await window.electronAPI.copyToClipboard(result.text);
            updateStatus('Transcription complete - Copied to clipboard!', 'success');
        } else {
            updateStatus('No speech detected', 'info');
        }
        
    } catch (error) {
        if (error.message.includes('Failed to fetch')) {
            updateStatus('Connection to Python server lost. Please check if it is still running.', 'error');
        } else {
            updateStatus('Transcription failed: ' + error.message, 'error');
        }
        console.error('Transcription error:', error);
    } finally {
        loadingDiv.style.display = 'none';
    }
}

// Append transcription
function appendTranscription(text) {
    const timestamp = new Date().toLocaleTimeString();
    const newText = `[${timestamp}] ${text}\n\n`;
    transcriptionArea.value += newText;
    transcriptionArea.scrollTop = transcriptionArea.scrollHeight;
}

// Copy all text
async function copyAll() {
    const text = transcriptionArea.value.trim();
    if (text) {
        await window.electronAPI.copyToClipboard(text);
        updateStatus('All text copied to clipboard!', 'success');
    }
}

// Clear text
function clearText() {
    transcriptionArea.value = '';
    updateStatus('Text cleared', 'info');
}

// Audio visualization
function visualizeAudio() {
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    function draw() {
        animationId = requestAnimationFrame(draw);
        
        analyser.getByteFrequencyData(dataArray);
        
        // Clear canvas
        canvasCtx.fillStyle = '#f0f0f0';
        canvasCtx.fillRect(0, 0, audioCanvas.width, audioCanvas.height);
        
        // Draw frequency bars
        const barWidth = (audioCanvas.width / bufferLength) * 2.5;
        let barHeight;
        let x = 0;
        
        // Calculate average volume
        let sum = 0;
        for (let i = 0; i < bufferLength; i++) {
            sum += dataArray[i];
        }
        const average = sum / bufferLength;
        const volumePercent = Math.round((average / 255) * 100);
        if (audioLevelSpan) audioLevelSpan.textContent = `Audio Level: ${volumePercent}%`;
        
        // Draw bars
        for (let i = 0; i < bufferLength; i++) {
            barHeight = (dataArray[i] / 255) * audioCanvas.height;
            
            // Color based on volume
            const hue = 120 - (dataArray[i] / 255) * 120; // Green to red
            canvasCtx.fillStyle = `hsl(${hue}, 70%, 50%)`;
            
            canvasCtx.fillRect(x, audioCanvas.height - barHeight, barWidth, barHeight);
            
            x += barWidth + 1;
        }
    }
    
    draw();
}

// Event listeners
recordBtn.addEventListener('click', toggleRecording);
copyBtn.addEventListener('click', copyAll);
clearBtn.addEventListener('click', clearText);

// Initialize on load
init();