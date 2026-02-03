// Initialize FullCalendar
document.addEventListener('DOMContentLoaded', function() {
    const calendarEl = document.getElementById('calendar');

    const calendar = new FullCalendar.Calendar(calendarEl, {
        initialView: 'dayGridMonth',
        headerToolbar: {
            left: 'prev,next',
            center: 'title',
            right: 'today'
        },
        height: 'auto',
        events: [
            // Sample events - these will be populated by the backend later
            {
                title: 'Garage Booking',
                start: '2026-01-25T10:00:00',
                end: '2026-01-25T12:00:00',
                color: '#ff8e3a'
            }
        ],
        eventClick: function(info) {
            alert('Event: ' + info.event.title + '\nTime: ' + info.event.start.toLocaleString());
        },
        dateClick: function(info) {
            console.log('Clicked on: ' + info.dateStr);
        },
        // Disable view switching
        viewClassNames: ['month-view-only']
    });

    calendar.render();

    // Initialize Lottie Animation
    const animation = lottie.loadAnimation({
        container: document.getElementById('lottie-animation'),
        renderer: 'svg',
        loop: true,
        autoplay: true,
        path: 'blob.json' // Path to your blob.json file
    });

    // ============ CONTINUOUS VOICE ASSISTANT LOGIC ============

    // State management
    let systemEnabled = false;
    let currentState = 'idle'; // idle, recording, processing
    let audioContext = null;
    let mediaStream = null;
    let mediaRecorder = null;
    let audioChunks = [];
    let silenceTimeout = null;
    let analyser = null;
    let vadAnimationFrame = null;
    let recordingMode = false; // Toggle for saving recordings

    // VAD configuration
    const VAD_THRESHOLD = 0.02; // Voice activity threshold
    const SILENCE_DURATION = 1500; // ms of silence before stopping
    const SPEECH_START_THRESHOLD = 3; // Consecutive frames above threshold to start recording

    // Get DOM elements
    const lottieContainer = document.getElementById('lottie-container');
    const recordingToggle = document.getElementById('recording-toggle');
    const recordingStatus = document.getElementById('recording-status');
    const toggleText = recordingToggle.querySelector('.toggle-text');

    let speechFrameCount = 0;

    // Recording toggle button handler
    recordingToggle.addEventListener('click', function() {
        recordingMode = !recordingMode;

        if (recordingMode) {
            recordingToggle.classList.add('active');
            toggleText.textContent = 'Recording ON';
            recordingStatus.textContent = 'Saving conversations to /recordings';
            recordingStatus.classList.add('active');
            console.log('Recording mode enabled - audio will be saved');
        } else {
            recordingToggle.classList.remove('active');
            toggleText.textContent = 'Recording OFF';
            recordingStatus.textContent = '';
            recordingStatus.classList.remove('active');
            console.log('Recording mode disabled');
        }
    });

    // Update UI state using blob animation
    function updateState(state) {
        currentState = state;

        // Remove all state classes
        lottieContainer.classList.remove('idle', 'recording', 'processing');

        switch(state) {
            case 'idle':
                // Pause animation, make blob smaller and dimmer
                animation.pause();
                lottieContainer.classList.add('idle');
                console.log('State: Idle - waiting for speech');
                break;
            case 'recording':
                // Play animation, make blob bigger and brighter
                animation.play();
                animation.setSpeed(1.2);
                lottieContainer.classList.add('recording');
                console.log('State: Recording - listening to speech');
                break;
            case 'processing':
                // Keep playing, slightly smaller, faster animation
                animation.play();
                animation.setSpeed(1.8);
                lottieContainer.classList.add('processing');
                console.log('State: Processing - transcribing audio');
                break;
            case 'error':
                // Pause on error
                animation.pause();
                lottieContainer.classList.add('idle');
                console.error('State: Error');
                break;
        }
    }

    // Continuous Voice Activity Detection
    function startContinuousVAD() {
        if (!analyser || !systemEnabled) return;

        const dataArray = new Uint8Array(analyser.frequencyBinCount);

        function detectVoice() {
            if (!systemEnabled) {
                vadAnimationFrame = null;
                return;
            }

            analyser.getByteTimeDomainData(dataArray);

            // Calculate RMS (Root Mean Square) for voice detection
            let sum = 0;
            for (let i = 0; i < dataArray.length; i++) {
                const normalized = (dataArray[i] - 128) / 128;
                sum += normalized * normalized;
            }
            const rms = Math.sqrt(sum / dataArray.length);

            // Voice detected
            if (rms > VAD_THRESHOLD) {
                speechFrameCount++;

                // Start recording if we detect speech and not already recording
                if (speechFrameCount >= SPEECH_START_THRESHOLD && currentState === 'idle') {
                    startRecording();
                }

                // Clear silence timeout if voice is detected during recording
                if (silenceTimeout && currentState === 'recording') {
                    clearTimeout(silenceTimeout);
                    silenceTimeout = null;
                }
            } else {
                speechFrameCount = 0;

                // Silence detected during recording - start countdown
                if (!silenceTimeout && currentState === 'recording') {
                    silenceTimeout = setTimeout(() => {
                        stopRecording();
                    }, SILENCE_DURATION);
                }
            }

            vadAnimationFrame = requestAnimationFrame(detectVoice);
        }

        detectVoice();
    }

    // Initialize continuous listening system
    async function initializeContinuousListening() {
        try {
            updateState('idle');

            // Request microphone access
            mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    sampleRate: 16000,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });

            // Create audio context for VAD
            audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: 16000
            });

            // Setup analyser for VAD
            analyser = audioContext.createAnalyser();
            analyser.fftSize = 2048;
            analyser.smoothingTimeConstant = 0.8;

            const source = audioContext.createMediaStreamSource(mediaStream);
            source.connect(analyser);

            systemEnabled = true;

            // Start continuous VAD
            startContinuousVAD();

            console.log('Continuous listening initialized');

        } catch (error) {
            console.error('Error initializing continuous listening:', error);
            updateState('error');
            systemEnabled = false;
            alert('Could not access microphone. Please check permissions.');
        }
    }

    // Start recording when speech is detected
    function startRecording() {
        if (currentState !== 'idle') return;

        try {
            // Setup MediaRecorder
            mediaRecorder = new MediaRecorder(mediaStream, {
                mimeType: 'audio/webm;codecs=opus'
            });

            audioChunks = [];

            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunks.push(event.data);
                }
            };

            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                sendAudioToBackend(audioBlob);
            };

            // Start recording
            mediaRecorder.start();
            updateState('recording');

            console.log('Recording started');

        } catch (error) {
            console.error('Error starting recording:', error);
        }
    }

    // Stop recording when silence is detected
    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state === 'recording') {
            mediaRecorder.stop();
            updateState('processing');
            console.log('Recording stopped');
        }

        if (silenceTimeout) {
            clearTimeout(silenceTimeout);
            silenceTimeout = null;
        }
    }

    // Send audio to backend
    function sendAudioToBackend(audioBlob) {
        console.log('Audio blob size:', audioBlob.size, 'bytes');

        if (recordingMode) {
            console.log('Sending with recording mode enabled');
        }

        const reader = new FileReader();
        reader.onloadend = () => {
            const base64Audio = reader.result.split(',')[1];
            socket.emit('audio_data', {
                audio: base64Audio,
                format: 'webm',
                recording_mode: recordingMode
            });
        };
        reader.readAsDataURL(audioBlob);
    }

    // Get transcription display element
    const transcriptionDisplay = document.getElementById('transcription-display');

    // Socket.IO connection
    const socket = io('http://localhost:5001');

    socket.on('connect', function() {
        console.log('Connected to backend server');
    });

    socket.on('disconnect', function() {
        console.log('Disconnected from backend server');
    });

    socket.on('bot_response', function(data) {
        console.log('User said:', data.user_text);
        console.log('Bot responded:', data.bot_text);

        // Log latency information
        if (data.latency_ms) {
            console.log(`⏱️ Backend Latency: ${data.latency_ms.backend}ms | Average: ${data.latency_ms.average}ms`);
        }

        // Check if recording was saved
        if (data.recorded) {
            console.log('✓ Audio saved to recordings directory');
        }

        // Display bot response
        let recordingBadge = '';
        if (data.recorded) {
            recordingBadge = '<span style="display: inline-block; margin-left: 10px; padding: 4px 10px; background-color: rgba(255, 142, 58, 0.2); border-radius: 12px; font-size: 0.75rem; color: #ff8e3a;">🔴 Recorded</span>';
        }

        // Display latency information
        let latencyInfo = '';
        if (data.latency_ms) {
            latencyInfo = `
                <div style="margin-top: 15px; padding: 10px; background-color: rgba(0, 0, 0, 0.3); border-radius: 8px; font-size: 0.85rem;">
                    <div style="color: rgba(255, 255, 255, 0.5); margin-bottom: 5px;">⏱️ Backend Latency:</div>
                    <div style="color: #ff8e3a;">
                        <span style="font-weight: 600;">${data.latency_ms.backend}ms</span> this response
                        ${data.latency_ms.average ? `| <span style="font-weight: 600;">${data.latency_ms.average}ms</span> average` : ''}
                    </div>
                    <div style="color: rgba(255, 255, 255, 0.4); font-size: 0.8rem; margin-top: 5px;">
                        TTS handled by ElevenLabs (client-side)
                    </div>
                </div>
            `;
        }

        transcriptionDisplay.innerHTML = `
            <div style="margin-bottom: 10px; color: rgba(255, 255, 255, 0.6); font-size: 0.9rem;">
                You: ${data.user_text}
            </div>
            <div style="color: #ff8e3a;">
                Assistant: ${data.bot_text}
                ${recordingBadge}
            </div>
            ${latencyInfo}
        `;
        transcriptionDisplay.classList.add('updating');

        // Remove updating class after animation
        setTimeout(() => {
            transcriptionDisplay.classList.remove('updating');
        }, 500);

        // Generate and play TTS using ElevenLabs via Puter.js
        console.log('Generating ElevenLabs TTS...');

        // Keep processing state while generating and playing audio
        updateState('processing');

        puter.ai.txt2speech(data.bot_text, {
            provider: "elevenlabs",
            voice: "21m00Tcm4TlvDq8ikWAM", // Rachel voice
            model: "eleven_flash_v2_5" // Fast generation model
        })
        .then(audio => {
            console.log('ElevenLabs audio ready, playing...');

            // Play the audio
            audio.play();

            // Return to idle when audio finishes
            audio.onended = () => {
                updateState('idle');
                console.log('ElevenLabs TTS playback finished');
            };
        })
        .catch(error => {
            console.error('Error generating/playing ElevenLabs TTS:', error);
            updateState('idle');
        });
    });

    socket.on('error', function(data) {
        console.error('Error from backend:', data.message);
        transcriptionDisplay.textContent = 'Error: ' + data.message;
        transcriptionDisplay.style.borderColor = '#ff4444';

        // Return to idle state
        setTimeout(() => {
            updateState('idle');
            transcriptionDisplay.style.borderColor = '';
        }, 3000);
    });

    socket.on('status', function(data) {
        console.log('Status:', data.message);
    });

    // Auto-start continuous listening on page load
    initializeContinuousListening();
});
