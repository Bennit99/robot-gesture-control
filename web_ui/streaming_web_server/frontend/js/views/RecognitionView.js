import { stopStream, fetchHasControl } from '../api.js';

export function renderRecognitionView(container, navigateTo) {
    container.innerHTML = `
        <div class="flex flex-col h-full bg-black text-white p-2 sm:p-4 pb-0 relative">
            <div class="flex justify-end items-center w-full absolute top-4 right-0 px-4 z-20 pointer-events-none">
                <button id="stop-btn" class="btn px-6 py-2 bg-red-600 font-bold text-white hover:bg-red-700 pointer-events-auto">
                    <i class="fa-solid fa-stop"></i> Stop Session
                </button>
            </div>

            <!-- Video Stream & Gesture (Resizable) -->
            <div class="w-full bg-gray-900 border border-gray-800 flex flex-col items-center justify-center rounded-lg relative resize-y overflow-hidden min-h-[300px]" style="height: 65vh;">
                <!-- Video stream container to constrain absolute positioning to the video -->
                <div class="flex-1 w-full flex justify-center items-center bg-black overflow-hidden min-h-0">
                    <div class="relative w-full h-full max-w-full max-h-full flex justify-center items-center" style="aspect-ratio: 16/9; max-height: 100%; max-width: 100%; object-fit: contain;">
                        <img id="video-stream" src="${window.BACKEND_URL}/video_feed?t=${Date.now()}" alt="Live Camera Video" class="w-full h-full object-contain z-0" />
                        <canvas id="slider-canvas" class="absolute top-0 left-0 w-full h-full pointer-events-none z-10"></canvas>
                        <div id="fps-overlay" class="absolute bottom-3 right-4 z-20 text-green-400 hidden pointer-events-none lowercase">
                            -- FPS
                        </div>

                        <!-- Series State Container -->
                        <div id="series-state-container" class="absolute top-24 left-4 z-10 flex flex-col gap-2 pointer-events-none">
                        </div>

                        <!-- Action overlay (floating glassmorphic card) -->
                        <div id="action-overlay" class="absolute bottom-4 left-4 right-4 md:left-8 md:right-8 bg-gray-900/80 backdrop-blur-md border border-gray-700/50 rounded-lg p-4 text-white flex flex-col gap-2 transition-all duration-300 opacity-0 pointer-events-none z-20 shadow-xl">
                            <div class="flex items-center justify-between">
                                <div class="flex items-center gap-3">
                                    <div class="flex h-3 w-3 relative">
                                        <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                                        <span class="relative inline-flex rounded-full h-3 w-3 bg-green-500"></span>
                                    </div>
                                    <span id="action-name" class="font-extrabold text-lg tracking-wide uppercase text-green-400">Sit</span>
                                </div>
                                <span id="action-mode" class="text-xs font-semibold uppercase tracking-wider px-2 py-0.5 rounded bg-gray-800 border border-gray-700 text-gray-400">Simulation Mode</span>
                            </div>
                            <p id="action-desc" class="text-sm text-gray-300 leading-relaxed font-medium text-left">Spot is sitting down</p>
                            <!-- Progress bar -->
                            <div class="w-full bg-gray-950 rounded-full h-2.5 overflow-hidden border border-gray-800 mt-1">
                                <div id="action-progress" class="bg-gradient-to-r from-green-500 to-emerald-400 h-full rounded-full transition-all duration-200" style="width: 0%;"></div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Gesture display -->
                <div id="gesture-overlay" class="w-full bg-gray-950 border-t border-gray-800 flex flex-col shrink-0 shadow-inner relative z-10 min-h-[5rem]">
                    <!-- Dynamic Gesture (Top / Center) -->
                    <div id="dynamic-gesture-text" class="text-green-400 font-bold text-center text-3xl sm:text-4xl tracking-widest uppercase py-2">
                        -
                    </div>
                    
                    <!-- Static Gestures (Bottom Split) -->
                    <div class="flex w-full border-t border-gray-800 text-sm font-semibold tracking-wider uppercase">
                        <div class="flex-1 flex flex-col items-center justify-center p-1 border-r border-gray-800">
                            <span class="text-gray-500 text-[10px] mb-0.5">Left Hand (Static)</span>
                            <span id="left-gesture-text" class="text-cyan-400">-</span>
                        </div>
                        <div class="flex-1 flex flex-col items-center justify-center p-1">
                            <span class="text-gray-500 text-[10px] mb-0.5">Right Hand (Static)</span>
                            <span id="right-gesture-text" class="text-cyan-400">-</span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Dashboard Drawer -->
            <div class="flex-1 w-full flex flex-col md:flex-row gap-4 mt-4 text-gray-800 pb-4 shrink-0 overflow-y-auto">
                <div class="md:w-1/2 bg-gray-50 dark:bg-gray-800 p-4 border border-gray-200 dark:border-gray-700 rounded-lg shadow overflow-y-auto w-full transition-colors duration-200">
                    <h3 class="text-lg font-bold mb-2 dark:text-gray-200">Live Statistics</h3>
                    <div id="stats-display" class="text-sm space-y-1 dark:text-gray-300">Gathering run statistics...</div>
                </div>
                
                <div class="md:w-1/2 bg-gray-50 dark:bg-gray-800 p-4 border border-gray-200 dark:border-gray-700 rounded-lg shadow overflow-y-auto w-full transition-colors duration-200">
                    <h3 class="text-lg font-bold mb-2 dark:text-gray-200">Gesture Distribution</h3>
                    <div id="gesture-display" class="text-sm space-y-1 dark:text-gray-300">N/A</div>
                </div>
            </div>
        </div>
    `;

    const stopBtn = container.querySelector('#stop-btn');
    const statsDiv = container.querySelector('#stats-display');
    const gestureDiv = container.querySelector('#gesture-display');

    const dynamicGestureText = container.querySelector('#dynamic-gesture-text');
    const leftGestureText = container.querySelector('#left-gesture-text');
    const rightGestureText = container.querySelector('#right-gesture-text');
    const fpsOverlay = container.querySelector('#fps-overlay');

    const actionOverlay = container.querySelector('#action-overlay');
    const actionName = container.querySelector('#action-name');
    const actionDesc = container.querySelector('#action-desc');
    const actionMode = container.querySelector('#action-mode');
    const actionProgress = container.querySelector('#action-progress');

    const sliderCanvas = container.querySelector('#slider-canvas');
    const ctx = sliderCanvas.getContext('2d');

    // Resize canvas to match its display size exactly
    const resizeObserver = new ResizeObserver(entries => {
        for (let entry of entries) {
            sliderCanvas.width = entry.contentRect.width;
            sliderCanvas.height = entry.contentRect.height;
        }
    });
    resizeObserver.observe(sliderCanvas);

    const seriesStateContainer = container.querySelector('#series-state-container');

    const getGestureIcon = (g) => {
        const m = {
            'Open_Palm': 'fa-hand',
            'Closed_Fist': 'fa-hand-fist',
            'Pointing_Up': 'fa-hand-pointer',
            'Thumb_Up': 'fa-thumbs-up',
            'Thumb_Down': 'fa-thumbs-down',
            'Victory': 'fa-hand-peace',
            'ILoveYou': 'fa-hand-love-you'
        };
        return m[g] || 'fa-hand';
    };

    let isStopping = false;
    let isSimulation = false;
    let activeXhr = null;

    // Fetch simulation mode to show correct indicator badge
    fetchHasControl().then(res => {
        if (res) {
            isSimulation = !res.has_control;
        }
    });

    const handleStop = async () => {
        if (isStopping) return;
        isStopping = true;
        stopBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Stopping...';
        stopBtn.classList.add('opacity-80', 'cursor-not-allowed');
        await stopStream(); // The application polling loop will automatically pick up state transitions
    };

    stopBtn.addEventListener('click', handleStop);

    // Initialize Metadata stream via XHR
    const xhr = new XMLHttpRequest();
    activeXhr = xhr;
    xhr.open('GET', `${window.BACKEND_URL}/metadata_feed`, true);
    xhr.responseType = 'text';

    xhr.onreadystatechange = function () {
        if (xhr.readyState === XMLHttpRequest.LOADING) {
            const responseText = xhr.responseText;
            const frames = responseText.split('--metadata');
            if (frames.length === 0) return;
            const lastFrame = frames[frames.length - 1];

            const gestureStringMatch = lastFrame.match(/Gesture-String: (.*)\r\n/);
            const leftStaticMatch = lastFrame.match(/Left-Static: (.*)\r\n/);
            const rightStaticMatch = lastFrame.match(/Right-Static: (.*)\r\n/);
            const seriesDataMatch = lastFrame.match(/Series-Data: (.*)\r\n/);
            const actionMatch = lastFrame.match(/Active-Action: (.*)\r\n/);
            const fpsMatch = lastFrame.match(/FPS: (.*)\r\n/);
            const statsMatch = lastFrame.match(/Stats: (.*)\r\n/);
            const sliderMatch = lastFrame.match(/Slider-Data: (.*)\r\n/);

            let fullGestureString = "";
            let leftStaticString = "NONE";
            let rightStaticString = "NONE";
            let seriesData = null;
            let fps = "";
            let sliderData = null;

            if (gestureStringMatch && gestureStringMatch[1].trim()) {
                fullGestureString = gestureStringMatch[1].trim();
            }
            if (leftStaticMatch && leftStaticMatch[1].trim()) {
                leftStaticString = leftStaticMatch[1].trim();
            }
            if (rightStaticMatch && rightStaticMatch[1].trim()) {
                rightStaticString = rightStaticMatch[1].trim();
            }
            if (seriesDataMatch && seriesDataMatch[1].trim()) {
                try {
                    seriesData = JSON.parse(seriesDataMatch[1]);
                } catch (e) {}
            }
            if (fpsMatch) {
                fps = fpsMatch[1];
            }
            if (sliderMatch && sliderMatch[1].trim()) {
                try {
                    sliderData = JSON.parse(sliderMatch[1]);
                } catch (e) {}
            }

            dynamicGestureText.innerText = fullGestureString || "-";
            leftGestureText.innerText = leftStaticString;
            rightGestureText.innerText = rightStaticString;

            fpsOverlay.style.display = 'block';
            fpsOverlay.innerText = `${fps} FPS`;

            if (seriesData && seriesData.active_series && seriesData.active_series.length > 0) {
                const percent = Math.max(0, (seriesData.remaining_ms / seriesData.timeout_ms) * 100);
                const dashoffset = `${100 - percent}`;

                seriesStateContainer.innerHTML = '';
                
                seriesData.active_series.forEach(series => {
                    const row = document.createElement('div');
                    row.className = "flex items-center gap-3 bg-gray-900/80 backdrop-blur-md border border-yellow-700/50 rounded-lg p-2 px-3 shadow-xl text-yellow-400";
                    
                    series.gestures.forEach((g, index) => {
                        if (index > 0) {
                            const arrow = document.createElement('i');
                            arrow.className = "fa-solid fa-arrow-right text-yellow-700/80 text-sm";
                            row.appendChild(arrow);
                        }

                        if (index < series.completed_count) {
                            // Completed gesture
                            const i = document.createElement('i');
                            i.className = `fa-solid ${getGestureIcon(g)} text-2xl drop-shadow-md text-yellow-400`;
                            row.appendChild(i);
                        } else if (index === series.completed_count) {
                            // Current expecting gesture (inside timer circle)
                            const circleContainer = document.createElement('div');
                            circleContainer.className = "relative flex items-center justify-center w-10 h-10";
                            circleContainer.innerHTML = `
                                <svg class="absolute w-full h-full -rotate-90" viewBox="0 0 36 36">
                                    <circle cx="18" cy="18" r="16" fill="none" stroke="rgba(202, 138, 4, 0.2)" stroke-width="3"></circle>
                                    <circle cx="18" cy="18" r="16" fill="none" stroke="#eab308" stroke-width="3" stroke-dasharray="100 100" stroke-dashoffset="${dashoffset}" stroke-linecap="round" class="transition-all duration-100 ease-linear"></circle>
                                </svg>
                                <div class="flex items-center justify-center absolute gap-1">
                                    <i class="fa-solid ${getGestureIcon(g)} text-lg drop-shadow-sm text-yellow-400"></i>
                                </div>
                            `;
                            row.appendChild(circleContainer);
                        } else {
                            // Future expected gesture (faded)
                            const i = document.createElement('i');
                            i.className = `fa-solid ${getGestureIcon(g)} text-xl drop-shadow-sm text-yellow-400 opacity-40`;
                            row.appendChild(i);
                        }
                    });

                    seriesStateContainer.appendChild(row);
                });
            } else {
                seriesStateContainer.innerHTML = '';
            }

            // Parse and update action overlay HUD
            if (actionMatch) {
                try {
                    const activeAction = JSON.parse(actionMatch[1]);
                    if (activeAction) {
                        actionName.innerText = activeAction.display_name;
                        actionDesc.innerText = activeAction.description;
                        actionMode.innerText = isSimulation ? "Simulation Mode" : "Live Mode";
                        if (isSimulation) {
                            actionMode.className = "text-xs font-semibold uppercase tracking-wider px-2 py-0.5 rounded bg-yellow-950/40 border border-yellow-700/50 text-yellow-400";
                        } else {
                            actionMode.className = "text-xs font-semibold uppercase tracking-wider px-2 py-0.5 rounded bg-emerald-950/40 border border-emerald-700/50 text-emerald-400";
                        }

                        // Progress calculation
                        const progressPercent = Math.min(100, Math.max(0, (activeAction.elapsed / activeAction.duration) * 100));
                        actionProgress.style.width = `${progressPercent}%`;

                        // Show overlay HUD
                        actionOverlay.classList.remove('opacity-0', 'pointer-events-none');
                        actionOverlay.classList.add('opacity-100', 'pointer-events-auto');
                    } else {
                        // Hide overlay HUD
                        actionOverlay.classList.remove('opacity-100', 'pointer-events-auto');
                        actionOverlay.classList.add('opacity-0', 'pointer-events-none');
                        actionProgress.style.width = `0%`;
                    }
                } catch (e) {
                    console.error('Failed to parse active action payload:', e);
                }
            } else {
                actionOverlay.classList.remove('opacity-100', 'pointer-events-auto');
                actionOverlay.classList.add('opacity-0', 'pointer-events-none');
            }

            // Draw Slider
            ctx.clearRect(0, 0, sliderCanvas.width, sliderCanvas.height);
            if (sliderData && sliderData.lx !== undefined) {
                const w = sliderCanvas.width;
                const h = sliderCanvas.height;
                const img = container.querySelector('#video-stream');
                
                // Calculate letterboxing to get the actual rendered image area
                let drawW = w;
                let drawH = h;
                let offsetX = 0;
                let offsetY = 0;

                if (img && img.naturalWidth && img.naturalHeight) {
                    const imgRatio = img.naturalWidth / img.naturalHeight;
                    const canvasRatio = w / h;
                    
                    if (imgRatio > canvasRatio) {
                        drawH = w / imgRatio;
                        offsetY = (h - drawH) / 2;
                    } else {
                        drawW = h * imgRatio;
                        offsetX = (w - drawW) / 2;
                    }
                }

                // Map normalized coords to the rendered image area
                const lx = offsetX + sliderData.lx * drawW;
                const ly = offsetY + sliderData.ly * drawH;
                const rx = offsetX + sliderData.rx * drawW;
                const ry = offsetY + sliderData.ry * drawH;
                const dx = rx - lx;
                const dy = ry - ly;

                const baseLineWidth = 12;

                // Base line
                ctx.beginPath();
                ctx.moveTo(lx, ly);
                ctx.lineTo(rx, ry);
                ctx.strokeStyle = 'rgba(200, 200, 200, 0.8)';
                ctx.lineWidth = baseLineWidth;
                ctx.stroke();

                // Fading extension line (right side)
                const extensionSteps = 10;
                for (let i = 0; i < extensionSteps; i++) {
                    const alpha = 1.0 - (i / extensionSteps);
                    ctx.beginPath();
                    ctx.moveTo(rx + dx * (i / extensionSteps), ry + dy * (i / extensionSteps));
                    ctx.lineTo(rx + dx * ((i + 1) / extensionSteps), ry + dy * ((i + 1) / extensionSteps));
                    ctx.strokeStyle = `rgba(255, 255, 0, ${alpha})`;
                    ctx.lineWidth = Math.max(2, baseLineWidth * alpha);
                    ctx.stroke();
                }

                // Fading extension line (left side)
                for (let i = 0; i < extensionSteps; i++) {
                    const alpha = 1.0 - (i / extensionSteps);
                    ctx.beginPath();
                    ctx.moveTo(lx - dx * (i / extensionSteps), ly - dy * (i / extensionSteps));
                    ctx.lineTo(lx - dx * ((i + 1) / extensionSteps), ly - dy * ((i + 1) / extensionSteps));
                    ctx.strokeStyle = `rgba(255, 255, 0, ${alpha})`;
                    ctx.lineWidth = Math.max(2, baseLineWidth * alpha);
                    ctx.stroke();
                }

                // Anchors
                ctx.beginPath();
                ctx.arc(lx, ly, 12, 0, 2 * Math.PI);
                ctx.fillStyle = '#10b981'; // green-500
                ctx.fill();

                ctx.beginPath();
                ctx.arc(rx, ry, 16, 0, 2 * Math.PI);
                ctx.fillStyle = '#facc15'; // yellow-400
                ctx.fill();

                // Text
                const text = `${sliderData.param}: ${sliderData.val.toFixed(2)}`;
                ctx.font = 'bold 20px "Inter", sans-serif';
                const textMetrics = ctx.measureText(text);
                const textWidth = textMetrics.width;
                const textHeight = 20;
                
                ctx.fillStyle = 'rgba(17, 24, 39, 0.9)'; // gray-900
                ctx.roundRect ? ctx.roundRect(lx, ly - 40, textWidth + 16, textHeight + 12, 6) : ctx.fillRect(lx, ly - 40, textWidth + 16, textHeight + 12);
                ctx.fill();
                
                ctx.fillStyle = '#f8fafc'; // slate-50
                ctx.fillText(text, lx + 8, ly - 23);
            }

            if (statsMatch) {
                try {
                    const stats = JSON.parse(statsMatch[1]);
                    const sortedGestures = Object.entries(stats.gesture_distribution)
                        .sort((a, b) => a[0].localeCompare(b[0]));

                    const gestureDistributionHtml = sortedGestures.map(([gesture, count]) =>
                        `<div class="flex justify-between border-b pb-1"><span>${gesture}</span> <span class="font-bold">${count}</span></div>`
                    ).join('');

                    statsDiv.innerHTML = `
                        <p>Total Frames: <b>${stats.total_frames}</b></p>
                        <p>Total Hands Detected: <b>${stats.total_hands_detected}</b></p>
                        <p>Total Gestures (excl. 'None'): <b>${stats.total_gestures_detected}</b></p>
                        <p>Avg Hands per Frame: <b>${stats.avg_hands_per_frame.toFixed(2)}</b></p>
                        <p>Avg Gesture Score: <b>${stats.avg_gesture_score.toFixed(2)}</b></p>
                    `;

                    gestureDiv.innerHTML = gestureDistributionHtml || 'No gestures yet';
                } catch (e) {
                    console.error('Failed to parse dynamic stat payload:', e);
                }
            }
        }
    };
    xhr.send();

    return () => {
        stopBtn.removeEventListener('click', handleStop);
        if (activeXhr) {
            activeXhr.abort();
            activeXhr = null;
        }
    };
}
