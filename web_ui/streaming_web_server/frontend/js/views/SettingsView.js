import { fetchSpotConfig, updateSpotConfig, fetchGestureConfig, updateGestureConfig, fetchHasControl, updateHasControl, fetchStaticGestureConfig, updateStaticGestureConfig, fetchTwoHandConfig, updateTwoHandConfig, BACKEND_URL } from '../api.js';

export function renderSettingsView(container, onClose) {
    container.innerHTML = `
        <div class="flex flex-col h-full bg-gray-100 dark:bg-gray-800 items-center justify-center p-4 transition-colors duration-200 overflow-hidden">
            <div class="w-full max-w-4xl bg-white dark:bg-gray-900 rounded-lg shadow-xl relative transition-colors duration-200 flex flex-col" style="max-height: 90vh;">
                
                <!-- Fixed Header -->
                <div class="p-6 pb-4 border-b dark:border-gray-700 flex justify-between items-center shrink-0">
                    <h2 class="text-3xl font-bold text-gray-800 dark:text-white m-0 tracking-tight">Settings</h2>
                    <button id="close-settings-btn" class="btn w-10 h-10 bg-gray-100 hover:bg-gray-200 dark:bg-gray-800 dark:hover:bg-gray-700 text-gray-500 dark:text-gray-400">
                        <i class="fa-solid fa-xmark fa-lg"></i>
                    </button>
                </div>

                <!-- Scrollable Body -->
                <div class="p-6 overflow-y-auto flex-1 config-form custom-scrollbar">
                    <div class="flex flex-col gap-8">
                        <!-- General Settings -->
                        <details class="group/chapter bg-gray-50 dark:bg-gray-800/50 rounded-lg border border-gray-200 dark:border-gray-700 overflow-hidden" open>
                            <summary class="flex justify-between items-center cursor-pointer p-4 select-none bg-gray-100/50 dark:bg-gray-800/80 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
                                <h3 class="text-xl font-bold text-gray-800 dark:text-white m-0">Display</h3>
                                <span class="transition-transform duration-200 group-open/chapter:rotate-180">
                                    <i class="fa-solid fa-chevron-down text-gray-500"></i>
                                </span>
                            </summary>
                            <div class="p-4 border-t border-gray-200 dark:border-gray-700">
                                <ul class="space-y-4 text-sm text-gray-700 dark:text-gray-300">
                                    <li class="list-row">
                                        <span class="font-medium flex items-center gap-2">
                                            <i class="fa-solid fa-moon "></i> Dark Mode
                                        </span>
                                        <button id="theme-toggle-btn" class="relative inline-flex h-6 w-11 items-center rounded-full bg-gray-300 dark:bg-blue-600 transition-colors focus:outline-none">
                                            <span class="translate-x-1 dark:translate-x-6 inline-block h-4 w-4 transform rounded-full bg-white transition-transform"></span>
                                        </button>
                                    </li>
                                    <li class="list-row">
                                        <span class="font-medium flex items-center gap-2 group relative">
                                            <i class="fa-solid fa-eye"></i> Show Unmapped Gestures
                                            <i class="fa-solid fa-circle-info text-gray-400 ml-1 cursor-help"></i>
                                            <span class="absolute bottom-full left-1/2 -translate-x-1/2 w-48 p-2 bg-gray-800 text-white text-xs rounded opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-10 text-center mb-1">
                                                If enabled, gestures without an assigned action will still be displayed in the recognition UI. Applies to both dynamic and static gestures.
                                            </span>
                                        </span>
                                        <button id="show-unmapped-toggle" class="relative inline-flex h-6 w-11 items-center rounded-full bg-gray-300 dark:bg-gray-600 transition-colors focus:outline-none">
                                            <span class="translate-x-1 inline-block h-4 w-4 transform rounded-full bg-white transition-transform"></span>
                                        </button>
                                    </li>
                                    <li class="list-row">
                                        <span class="font-medium flex items-center gap-2 group relative">
                                            <i class="fa-solid fa-keyboard"></i> Web UI Manual Controls
                                            <i class="fa-solid fa-circle-info text-gray-400 ml-1 cursor-help"></i>
                                            <span class="absolute bottom-full left-1/2 -translate-x-1/2 w-48 p-2 bg-gray-800 text-white text-xs rounded opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-10 text-center mb-1">
                                                Enable or disable WASD and QE keyboard overrides across the web interface.
                                            </span>
                                        </span>
                                        <button id="manual-control-toggle" class="relative inline-flex h-6 w-11 items-center rounded-full bg-gray-300 dark:bg-gray-600 transition-colors focus:outline-none">
                                            <span class="translate-x-1 inline-block h-4 w-4 transform rounded-full bg-white transition-transform"></span>
                                        </button>
                                    </li>
                                    <li class="list-row">
                                        <div class="flex flex-col gap-1 w-full">
                                            <span class="font-medium flex items-center gap-2">
                                                <i class="fa-solid fa-server"></i> Operation Mode
                                            </span>
                                            <span id="sim-mode-status" class="text-xs text-gray-500 dark:text-gray-400">Loading...</span>
                                        </div>
                                        <button id="toggle-sim-mode-btn" class="btn px-3 py-1 bg-blue-500 hover:bg-blue-600 dark:bg-blue-600 dark:hover:bg-blue-700 text-white text-sm font-medium">
                                            Switch
                                        </button>
                                    </li>
                                </ul>
                            </div>
                        </details>

                        <!-- Spot Parameters -->
                        <details class="group/chapter bg-gray-50 dark:bg-gray-800/50 rounded-lg border border-gray-200 dark:border-gray-700 overflow-hidden" open>
                            <summary class="flex justify-between items-center cursor-pointer p-4 select-none bg-gray-100/50 dark:bg-gray-800/80 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
                                <h3 class="text-xl font-bold text-gray-800 dark:text-white m-0">Spot Parameters</h3>
                                <span class="transition-transform duration-200 group-open/chapter:rotate-180">
                                    <i class="fa-solid fa-chevron-down text-gray-500"></i>
                                </span>
                            </summary>
                            <div class="p-4 border-t border-gray-200 dark:border-gray-700">
                                <div class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4 text-sm text-gray-700 dark:text-gray-300">
                                    <div>
                                        <label class="block mb-1 font-medium text-gray-600 dark:text-gray-400 group relative">
                                            Velocity Base Speed (m/s)
                                            <i class="fa-solid fa-circle-info text-gray-400 ml-1 cursor-help"></i>
                                            <span class="absolute bottom-full left-1/2 -translate-x-1/2 w-48 p-2 bg-gray-800 text-white text-xs rounded opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-10 text-center mb-1">
                                                The maximum forward/backward walking speed of the robot.
                                            </span>
                                        </label>
                                        <input id="spot-vel-speed" type="number" step="0.1" class="input-field shadow-sm bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-600 custom-number-input">
                                    </div>
                                    <div>
                                        <label class="block mb-1 font-medium text-gray-600 dark:text-gray-400 group relative">
                                            Velocity Angular (rad/s)
                                            <i class="fa-solid fa-circle-info text-gray-400 ml-1 cursor-help"></i>
                                            <span class="absolute bottom-full left-1/2 -translate-x-1/2 w-48 p-2 bg-gray-800 text-white text-xs rounded opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-10 text-center mb-1">
                                                The turning speed (yaw rate) when the robot rotates left or right.
                                            </span>
                                        </label>
                                        <input id="spot-vel-angular" type="number" step="0.1" class="input-field shadow-sm bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-600 custom-number-input">
                                    </div>
                                    <div>
                                        <label class="block mb-1 font-medium text-gray-600 dark:text-gray-400 group relative">
                                            Command Duration (s)
                                            <i class="fa-solid fa-circle-info text-gray-400 ml-1 cursor-help"></i>
                                            <span class="absolute bottom-full left-1/2 -translate-x-1/2 w-48 p-2 bg-gray-800 text-white text-xs rounded opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-10 text-center mb-1">
                                                How long Spot executes a single movement command before stopping automatically.
                                            </span>
                                        </label>
                                        <input id="spot-cmd-duration" type="number" step="0.1" class="input-field shadow-sm bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-600 custom-number-input">
                                    </div>
                                    <div>
                                        <label class="block mb-1 font-medium text-gray-600 dark:text-gray-400 group relative">
                                            Body Pitch Angle (rad)
                                            <i class="fa-solid fa-circle-info text-gray-400 ml-1 cursor-help"></i>
                                            <span class="absolute bottom-full left-1/2 -translate-x-1/2 w-48 p-2 bg-gray-800 text-white text-xs rounded opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-10 text-center mb-1">
                                                The forward/backward tilt limit when using Look Up, Look Down, or Bow.
                                            </span>
                                        </label>
                                        <input id="spot-pitch-angle" type="number" step="0.1" class="input-field shadow-sm bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-600 custom-number-input">
                                    </div>
                                    <div>
                                        <label class="block mb-1 font-medium text-gray-600 dark:text-gray-400 group relative">
                                            Body Yaw Angle (rad)
                                            <i class="fa-solid fa-circle-info text-gray-400 ml-1 cursor-help"></i>
                                            <span class="absolute bottom-full left-1/2 -translate-x-1/2 w-48 p-2 bg-gray-800 text-white text-xs rounded opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-10 text-center mb-1">
                                                The left/right twist limit of Spot's body when using Look Left or Look Right while standing still.
                                            </span>
                                        </label>
                                        <input id="spot-yaw-angle" type="number" step="0.1" class="input-field shadow-sm bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-600 custom-number-input">
                                    </div>
                                </div>
                            </div>
                        </details>

                        <!-- Dynamic Gesture Mappings -->
                        <details class="group/chapter bg-gray-50 dark:bg-gray-800/50 rounded-lg border border-gray-200 dark:border-gray-700 overflow-hidden" open>
                            <summary class="flex justify-between items-center cursor-pointer p-4 select-none bg-gray-100/50 dark:bg-gray-800/80 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
                                <h3 class="text-xl font-bold text-gray-800 dark:text-white m-0">Dynamic Gestures</h3>
                                <span class="transition-transform duration-200 group-open/chapter:rotate-180">
                                    <i class="fa-solid fa-chevron-down text-gray-500"></i>
                                </span>
                            </summary>
                            <div class="p-4 border-t border-gray-200 dark:border-gray-700">
                                <div class="mb-4 flex items-center justify-between">
                                    <div class="w-3/4">
                                        <label class="block mb-1 text-sm font-medium text-gray-600 dark:text-gray-400">
                                            Confidence Threshold: <span id="dynamic-conf-val" class="font-bold">0.5</span>
                                        </label>
                                        <input id="dynamic-conf-slider" type="range" min="0.1" max="0.95" step="0.05" value="0.5" class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700">
                                    </div>
                                    <button id="reset-dynamic-btn" class="text-xs bg-red-100 text-red-600 dark:bg-red-900/30 dark:text-red-400 hover:bg-red-200 dark:hover:bg-red-900/50 px-2 py-1 rounded transition-colors focus:outline-none mt-4">
                                        <i class="fa-solid fa-rotate-left mr-1"></i> Reset All
                                    </button>
                                </div>
                                <div class="flex flex-col sm:flex-row items-start sm:items-center mb-4 border-b pb-2 dark:border-gray-700 gap-2">
                                    <h4 class="font-bold text-gray-700 dark:text-gray-300 w-full sm:w-1/2">Gesture</h4>
                                    <div class="w-full sm:w-1/2 flex items-center justify-between">
                                        <h4 class="font-bold text-gray-700 dark:text-gray-300 hidden sm:block">Action</h4>
                                    </div>
                                </div>
                                <div class="flex flex-col gap-3" id="gesture-mappings-container">
                                    <p class="text-gray-500 dark:text-gray-400 italic">Loading gestures...</p>
                                </div>
                                <button id="add-dynamic-mapping-btn" class="mt-4 btn px-3 py-1 bg-gray-200 hover:bg-gray-300 dark:bg-gray-700 dark:hover:bg-gray-600 text-gray-700 dark:text-gray-300 text-sm font-medium rounded hidden">
                                    <i class="fa-solid fa-plus mr-1"></i> Add Mapping
                                </button>
                            </div>
                        </details>
                        
                        <!-- Static Gesture Mappings -->
                        <details class="group/chapter bg-gray-50 dark:bg-gray-800/50 rounded-lg border border-gray-200 dark:border-gray-700 overflow-hidden" open>
                            <summary class="flex justify-between items-center cursor-pointer p-4 select-none bg-gray-100/50 dark:bg-gray-800/80 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
                                <h3 class="text-xl font-bold text-gray-800 dark:text-white m-0">Static Gestures</h3>
                                <span class="transition-transform duration-200 group-open/chapter:rotate-180">
                                    <i class="fa-solid fa-chevron-down text-gray-500"></i>
                                </span>
                            </summary>
                            <div class="p-4 border-t border-gray-200 dark:border-gray-700">
                                <div class="mb-4 flex items-center justify-between">
                                    <div class="w-3/4">
                                        <label class="block mb-1 text-sm font-medium text-gray-600 dark:text-gray-400">
                                            Confidence Threshold: <span id="static-conf-val" class="font-bold">0.6</span>
                                        </label>
                                        <input id="static-conf-slider" type="range" min="0.1" max="0.95" step="0.05" value="0.6" class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700">
                                    </div>
                                    <button id="reset-static-btn" class="text-xs bg-red-100 text-red-600 dark:bg-red-900/30 dark:text-red-400 hover:bg-red-200 dark:hover:bg-red-900/50 px-2 py-1 rounded transition-colors focus:outline-none mt-4">
                                        <i class="fa-solid fa-rotate-left mr-1"></i> Reset All
                                    </button>
                                </div>

                                <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
                                    <div>
                                        <h4 class="font-bold text-gray-700 dark:text-gray-300 mb-2 border-b dark:border-gray-700 pb-1">Left Hand</h4>
                                        <div class="flex flex-col gap-3" id="static-left-container">
                                            <p class="text-gray-500 dark:text-gray-400 italic text-sm">Loading...</p>
                                        </div>
                                        <button id="add-static-left-btn" class="mt-4 btn px-3 py-1 bg-gray-200 hover:bg-gray-300 dark:bg-gray-700 dark:hover:bg-gray-600 text-gray-700 dark:text-gray-300 text-sm font-medium rounded hidden">
                                            <i class="fa-solid fa-plus mr-1"></i> Add Left Mapping
                                        </button>
                                    </div>
                                    <div>
                                        <h4 class="font-bold text-gray-700 dark:text-gray-300 mb-2 border-b dark:border-gray-700 pb-1">Right Hand</h4>
                                        <div class="flex flex-col gap-3" id="static-right-container">
                                            <p class="text-gray-500 dark:text-gray-400 italic text-sm">Loading...</p>
                                        </div>
                                        <button id="add-static-right-btn" class="mt-4 btn px-3 py-1 bg-gray-200 hover:bg-gray-300 dark:bg-gray-700 dark:hover:bg-gray-600 text-gray-700 dark:text-gray-300 text-sm font-medium rounded hidden">
                                            <i class="fa-solid fa-plus mr-1"></i> Add Right Mapping
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </details>
                        
                        <!-- Two Hand Combinations -->
                        <details class="group/chapter bg-gray-50 dark:bg-gray-800/50 rounded-lg border border-gray-200 dark:border-gray-700 overflow-hidden" open>
                            <summary class="flex justify-between items-center cursor-pointer p-4 select-none bg-gray-100/50 dark:bg-gray-800/80 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
                                <h3 class="text-xl font-bold text-gray-800 dark:text-white m-0">Two-Hand Combinations</h3>
                                <span class="transition-transform duration-200 group-open/chapter:rotate-180">
                                    <i class="fa-solid fa-chevron-down text-gray-500"></i>
                                </span>
                            </summary>
                            <div class="p-4 border-t border-gray-200 dark:border-gray-700">
                                <div class="mb-4 flex items-center justify-between">
                                    <div class="w-3/4">
                                        <label class="block mb-1 text-sm font-medium text-gray-600 dark:text-gray-400 group relative">
                                            Second Hand Wait Timeout: <span id="two-hand-wait-val" class="font-bold">300</span> ms
                                            <i class="fa-solid fa-circle-info text-gray-400 ml-1 cursor-help"></i>
                                            <span class="absolute bottom-full left-0 w-64 p-2 bg-gray-800 text-white text-xs rounded opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-10 mb-1">
                                                When one hand performs a gesture that is part of a combination, wait this long for the second hand before firing the single-hand action.
                                            </span>
                                        </label>
                                        <input id="two-hand-wait-slider" type="range" min="0" max="1000" step="50" value="300" class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700">
                                    </div>
                                    <button id="reset-combo-btn" class="text-xs bg-red-100 text-red-600 dark:bg-red-900/30 dark:text-red-400 hover:bg-red-200 dark:hover:bg-red-900/50 px-2 py-1 rounded transition-colors focus:outline-none mt-4">
                                        <i class="fa-solid fa-rotate-left mr-1"></i> Reset All
                                    </button>
                                </div>

                                <div class="flex flex-col gap-3" id="two-hand-container">
                                    <p class="text-gray-500 dark:text-gray-400 italic">Loading combinations...</p>
                                </div>
                                
                                <button id="add-combo-btn" class="mt-4 btn px-3 py-1 bg-gray-200 hover:bg-gray-300 dark:bg-gray-700 dark:hover:bg-gray-600 text-gray-700 dark:text-gray-300 text-sm font-medium rounded">
                                    <i class="fa-solid fa-plus mr-1"></i> Add Combination
                                </button>
                            </div>
                        </details>

                        <!-- Slider Mappings (Two Hand) -->
                        <details class="group/chapter bg-gray-50 dark:bg-gray-800/50 rounded-lg border border-gray-200 dark:border-gray-700 overflow-hidden" open>
                            <summary class="flex justify-between items-center cursor-pointer p-4 select-none bg-gray-100/50 dark:bg-gray-800/80 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
                                <h3 class="text-xl font-bold text-gray-800 dark:text-white m-0">Slider</h3>
                                <span class="transition-transform duration-200 group-open/chapter:rotate-180">
                                    <i class="fa-solid fa-chevron-down text-gray-500"></i>
                                </span>
                            </summary>
                            <div class="p-4 border-t border-gray-200 dark:border-gray-700">
                                <div class="mb-4 flex items-center justify-between">
                                    <p class="text-sm text-gray-600 dark:text-gray-400">Map a two-hand combination to a float parameter to enter Slider Mode.</p>
                                    <button id="reset-slider-btn" class="text-xs bg-red-100 text-red-600 dark:bg-red-900/30 dark:text-red-400 hover:bg-red-200 dark:hover:bg-red-900/50 px-2 py-1 rounded transition-colors focus:outline-none mt-4">
                                        <i class="fa-solid fa-rotate-left mr-1"></i> Reset All
                                    </button>
                                </div>
                                <div class="flex flex-col gap-3" id="slider-container">
                                    <p class="text-gray-500 dark:text-gray-400 italic">Loading sliders...</p>
                                </div>
                                <button id="add-slider-btn" class="mt-4 btn px-3 py-1 bg-gray-200 hover:bg-gray-300 dark:bg-gray-700 dark:hover:bg-gray-600 text-gray-700 dark:text-gray-300 text-sm font-medium rounded">
                                    <i class="fa-solid fa-plus mr-1"></i> Add Slider Mapping
                                </button>
                            </div>
                        </details>
                        
                        <!-- Stacked Gesture -->
                        <details class="group/chapter bg-gray-50 dark:bg-gray-800/50 rounded-lg border border-gray-200 dark:border-gray-700 overflow-hidden" open>
                            <summary class="flex justify-between items-center cursor-pointer p-4 select-none bg-gray-100/50 dark:bg-gray-800/80 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
                                <h3 class="text-xl font-bold text-gray-800 dark:text-white m-0">Stacked Gesture</h3>
                                <span class="transition-transform duration-200 group-open/chapter:rotate-180">
                                    <i class="fa-solid fa-chevron-down text-gray-500"></i>
                                </span>
                            </summary>
                            <div class="p-4 border-t border-gray-200 dark:border-gray-700">
                                <div class="mb-4 flex items-center justify-between">
                                    <div class="w-3/4">
                                        <label class="block mb-1 text-sm font-medium text-gray-600 dark:text-gray-400 group relative">
                                            Series Timeout: <span id="series-timeout-val" class="font-bold">2000</span> ms
                                            <i class="fa-solid fa-circle-info text-gray-400 ml-1 cursor-help"></i>
                                            <span class="absolute bottom-full left-0 w-64 p-2 bg-gray-800 text-white text-xs rounded opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-10 mb-1">
                                                How long to wait for the next gesture in a series before timing out and resetting.
                                            </span>
                                        </label>
                                        <input id="series-timeout-slider" type="range" min="500" max="5000" step="100" value="2000" class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700">
                                    </div>
                                    <button id="reset-series-btn" class="text-xs bg-red-100 text-red-600 dark:bg-red-900/30 dark:text-red-400 hover:bg-red-200 dark:hover:bg-red-900/50 px-2 py-1 rounded transition-colors focus:outline-none mt-4">
                                        <i class="fa-solid fa-rotate-left mr-1"></i> Reset All
                                    </button>
                                </div>

                                <div class="flex flex-col gap-3" id="series-container">
                                    <p class="text-gray-500 dark:text-gray-400 italic">Loading series...</p>
                                </div>
                                
                                <button id="add-series-btn" class="mt-4 btn px-3 py-1 bg-gray-200 hover:bg-gray-300 dark:bg-gray-700 dark:hover:bg-gray-600 text-gray-700 dark:text-gray-300 text-sm font-medium rounded">
                                    <i class="fa-solid fa-plus mr-1"></i> Add Series
                                </button>
                            </div>
                        </details>
                    </div>
                </div>

                <!-- Fixed Footer -->
                <div class="p-6 pt-4 border-t dark:border-gray-700 shrink-0 bg-white dark:bg-gray-900 rounded-b-lg">
                    <div class="flex gap-4 w-full">
                        <button id="main-action-btn" class="btn flex-1 bg-gray-500 hover:bg-gray-600 text-white py-3 font-bold transition-all duration-300 ease-in-out">
                            Close
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `;

    const closeBtn = container.querySelector('#close-settings-btn');
    const mainBtn = container.querySelector('#main-action-btn');
    const themeToggleBtn = container.querySelector('#theme-toggle-btn');
    const manualControlToggleBtn = container.querySelector('#manual-control-toggle');
    const toggleSimModeBtn = container.querySelector('#toggle-sim-mode-btn');
    const simModeStatus = container.querySelector('#sim-mode-status');
    const mappingsContainer = container.querySelector('#gesture-mappings-container');

    // New static & two-hand elements
    const staticConfSlider = container.querySelector('#static-conf-slider');
    const staticConfVal = container.querySelector('#static-conf-val');
    const staticLeftContainer = container.querySelector('#static-left-container');
    const staticRightContainer = container.querySelector('#static-right-container');

    // Reset buttons
    const resetDynamicBtn = container.querySelector('#reset-dynamic-btn');
    const resetStaticBtn = container.querySelector('#reset-static-btn');
    const resetComboBtn = container.querySelector('#reset-combo-btn');

    const twoHandWaitSlider = container.querySelector('#two-hand-wait-slider');
    const twoHandWaitVal = container.querySelector('#two-hand-wait-val');
    const twoHandContainer = container.querySelector('#two-hand-container');
    const addComboBtn = container.querySelector('#add-combo-btn');

    let sliderCount = 0;
    const sliderContainer = container.querySelector('#slider-container');
    const addSliderBtn = container.querySelector('#add-slider-btn');
    const resetSliderBtn = container.querySelector('#reset-slider-btn');

    const seriesTimeoutSlider = container.querySelector('#series-timeout-slider');
    const seriesTimeoutVal = container.querySelector('#series-timeout-val');
    const seriesContainer = container.querySelector('#series-container');
    const addSeriesBtn = container.querySelector('#add-series-btn');
    const resetSeriesBtn = container.querySelector('#reset-series-btn');

    const formBody = container.querySelector('.config-form');

    let hasUnsavedChanges = false;
    let currentHasControl = null;
    let availableActions = ['None'];
    let comboCount = 0;
    let seriesCount = 0;
    let globalDynamicGesturesList = [];

    // Built-in MediaPipe static gestures
    const staticGesturesList = [
        'Closed_Fist', 'Open_Palm', 'Pointing_Up',
        'Thumb_Down', 'Thumb_Up', 'Victory', 'ILoveYou'
    ];

    const markChanged = () => {
        if (!hasUnsavedChanges) {
            hasUnsavedChanges = true;
            mainBtn.innerText = 'Save & Close';
            mainBtn.classList.remove('bg-gray-500', 'hover:bg-gray-600');
            mainBtn.classList.add('bg-blue-600', 'hover:bg-blue-700', 'dark:bg-blue-500', 'dark:hover:bg-blue-600');
        }
    };

    formBody.addEventListener('input', markChanged);
    formBody.addEventListener('change', markChanged);

    // Load config
    Promise.all([
        fetchSpotConfig(),
        fetchGestureConfig(),
        fetchHasControl(),
        fetchStaticGestureConfig(),
        fetchTwoHandConfig()
    ]).then(([spotConfig, gestureConfig, hasControlData, staticConfig, twoHandConfig]) => {

        // Load simulation mode
        if (hasControlData) {
            currentHasControl = hasControlData.has_control;
            updateSimModeStatus(currentHasControl);
        }

        if (spotConfig) {
            container.querySelector('#spot-vel-speed').value = spotConfig.VELOCITY_BASE_SPEED ?? 1;
            container.querySelector('#spot-vel-angular').value = spotConfig.VELOCITY_BASE_ANGULAR ?? 0.8;
            container.querySelector('#spot-cmd-duration').value = spotConfig.VELOCITY_CMD_DURATION ?? 1;
            container.querySelector('#spot-pitch-angle').value = spotConfig.BODY_PITCH_ANGLE ?? 0.4;
            container.querySelector('#spot-yaw-angle').value = spotConfig.BODY_YAW_ANGLE ?? 0.4;
        }

        if (gestureConfig) {
            availableActions = ['None', ...(gestureConfig.actions || [])];

            // Render Dynamic Gestures
            if (gestureConfig.mappings) {
                globalDynamicGesturesList = Object.keys(gestureConfig.mappings);
                const dynamicGesturesList = globalDynamicGesturesList;
                mappingsContainer.innerHTML = '';
                Object.entries(gestureConfig.mappings).forEach(([gesture, currentAction]) => {
                    if (currentAction && currentAction !== 'None') {
                        mappingsContainer.appendChild(createSingleMappingRow(dynamicGesturesList, gesture, currentAction, 'dynamic-mapping-select', false));
                    }
                });

                const addDynamicBtn = container.querySelector('#add-dynamic-mapping-btn');
                addDynamicBtn.classList.remove('hidden');
                addDynamicBtn.addEventListener('click', (e) => {
                    e.preventDefault();
                    mappingsContainer.appendChild(createSingleMappingRow(dynamicGesturesList, dynamicGesturesList[0], 'None', 'dynamic-mapping-select', false));
                    markChanged();
                });
            }

            if (gestureConfig.confidence_threshold !== undefined) {
                container.querySelector('#dynamic-conf-slider').value = gestureConfig.confidence_threshold;
                container.querySelector('#dynamic-conf-val').innerText = gestureConfig.confidence_threshold.toFixed(2);
            }

            const showUnmappedBtn = container.querySelector('#show-unmapped-toggle');
            updateToggleUI(showUnmappedBtn, gestureConfig.show_unmapped);
        }

        // Render Static Gestures
        if (staticConfig) {
            staticConfSlider.value = staticConfig.confidence_threshold;
            staticConfVal.innerText = staticConfig.confidence_threshold.toFixed(2);

            staticLeftContainer.innerHTML = '';
            staticGesturesList.forEach(gesture => {
                const action = staticConfig.mappings_left ? staticConfig.mappings_left[gesture] : null;
                if (action && action !== 'None') {
                    staticLeftContainer.appendChild(createSingleMappingRow(staticGesturesList, gesture, action, 'static-left-select', true));
                }
            });

            const addStaticLeftBtn = container.querySelector('#add-static-left-btn');
            addStaticLeftBtn.classList.remove('hidden');
            addStaticLeftBtn.addEventListener('click', (e) => {
                e.preventDefault();
                staticLeftContainer.appendChild(createSingleMappingRow(staticGesturesList, staticGesturesList[0], 'None', 'static-left-select', true));
                markChanged();
            });

            staticRightContainer.innerHTML = '';
            staticGesturesList.forEach(gesture => {
                const action = staticConfig.mappings_right ? staticConfig.mappings_right[gesture] : null;
                if (action && action !== 'None') {
                    staticRightContainer.appendChild(createSingleMappingRow(staticGesturesList, gesture, action, 'static-right-select', true));
                }
            });

            const addStaticRightBtn = container.querySelector('#add-static-right-btn');
            addStaticRightBtn.classList.remove('hidden');
            addStaticRightBtn.addEventListener('click', (e) => {
                e.preventDefault();
                staticRightContainer.appendChild(createSingleMappingRow(staticGesturesList, staticGesturesList[0], 'None', 'static-right-select', true));
                markChanged();
            });
        }

        // Render Two Hand Config
        if (twoHandConfig) {
            if (twoHandConfig.two_hand_wait_ms !== undefined) {
                twoHandWaitSlider.value = twoHandConfig.two_hand_wait_ms;
                twoHandWaitVal.innerText = twoHandConfig.two_hand_wait_ms;
            }
            if (twoHandConfig.combinations) {
                twoHandContainer.innerHTML = '';
                sliderContainer.innerHTML = '';
                Object.entries(twoHandConfig.combinations).forEach(([comboKey, currentAction]) => {
                    const parts = comboKey.split('+');
                    if (parts.length === 2) {
                        if (currentAction && currentAction.startsWith('config.')) {
                            sliderContainer.appendChild(createSliderRow(parts[0], parts[1], currentAction));
                        } else {
                            twoHandContainer.appendChild(createComboRow(parts[0], parts[1], currentAction));
                        }
                    }
                });
            }

            if (twoHandConfig.series_timeout_ms !== undefined) {
                seriesTimeoutSlider.value = twoHandConfig.series_timeout_ms;
                seriesTimeoutVal.innerText = twoHandConfig.series_timeout_ms;
            }
            if (twoHandConfig.series) {
                seriesContainer.innerHTML = '';
                Object.entries(twoHandConfig.series).forEach(([seriesKey, currentAction]) => {
                    // key is "L:Open_Palm,R:Closed_Fist"
                    const parts = seriesKey.split(',');
                    seriesContainer.appendChild(createSeriesRow(parts, currentAction));
                });
            }
        }
    });

    function getGestureIcon(gestureName) {
        const map = {
            'Closed_Fist': 'fa-hand-fist',
            'Open_Palm': 'fa-hand',
            'Pointing_Up': 'fa-hand-pointer',
            'Thumb_Down': 'fa-thumbs-down',
            'Thumb_Up': 'fa-thumbs-up',
            'Victory': 'fa-hand-peace',
            'ILoveYou': 'fa-hand-love-you'
        };
        return map[gestureName] || 'fa-hand-sparkles';
    }

    // No hardcoded paramActions anymore, we check if action starts with 'config.'


    function createGestureSelectEl(gestureList, selectedGesture, isStatic, selectClass, prefix = '') {
        const fragment = document.createDocumentFragment();
        
        const select = document.createElement('select');
        select.className = `input-field text-xs py-1 bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-600 ${selectClass}`;
        
        gestureList.forEach(g => {
            const opt = document.createElement('option');
            opt.value = g;
            opt.innerText = prefix ? `${prefix}${g.replace(/_/g, ' ')}` : g.replace(/_/g, ' ');
            if (g === selectedGesture) opt.selected = true;
            select.appendChild(opt);
        });

        if (isStatic) {
            const iconEl = document.createElement('i');
            iconEl.className = `fa-solid ${getGestureIcon(selectedGesture || gestureList[0])} text-yellow-500 w-4 text-center shrink-0`;
            select.addEventListener('change', (e) => {
                iconEl.className = `fa-solid ${getGestureIcon(e.target.value)} text-yellow-500 w-4 text-center shrink-0`;
            });
            fragment.appendChild(iconEl);
        }
        
        fragment.appendChild(select);
        return { fragment, select };
    }

    function createActionSelectEl(currentAction, selectClass, filterSlidersOnly = false) {
        const fragment = document.createDocumentFragment();

        let baseAction = currentAction;
        let actionParam = '25';
        if (currentAction && currentAction.includes('(')) {
            const match = currentAction.match(/^(.*?)\((.*?)\)$/);
            if (match) {
                baseAction = match[1];
                actionParam = match[2];
            }
        }

        const actionSelect = document.createElement('select');
        actionSelect.className = `input-field w-full py-1 ${selectClass} bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-600`;

        availableActions.forEach(a => {
            if (filterSlidersOnly && !a.startsWith('config.')) return;

            const opt = document.createElement('option');
            opt.value = a;
            opt.className = 'text-gray-800 dark:text-gray-200 not-italic';
            let displayName = a;
            let group = 'none';

            if (a.startsWith('spot.')) {
                displayName = a.replace('spot.', '').split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
                group = 'movement';
            } else if (a.startsWith('config.')) {
                displayName = a.replace('config.', '').split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase()).join(' ');
                group = 'parameter';
            }
            opt.innerText = displayName;

            if (a === baseAction || (!baseAction && a === 'None')) opt.selected = true;

            opt._group = group;
            actionSelect._options = actionSelect._options || [];
            actionSelect._options.push(opt);
        });

        if (filterSlidersOnly) {
            const noneOpt = document.createElement('option');
            noneOpt.value = 'None';
            noneOpt.innerText = 'None';
            if (baseAction === 'None' || !baseAction) noneOpt.selected = true;
            noneOpt._group = 'none';
            actionSelect._options = actionSelect._options || [];
            actionSelect._options.push(noneOpt);
        }

        const noneGroup = document.createDocumentFragment();
        const movementGroup = document.createElement('optgroup');
        movementGroup.label = 'Spot Movement';
        const parameterGroup = document.createElement('optgroup');
        parameterGroup.label = 'Parameters';

        (actionSelect._options || []).forEach(opt => {
            if (opt._group === 'movement') movementGroup.appendChild(opt);
            else if (opt._group === 'parameter') parameterGroup.appendChild(opt);
            else noneGroup.appendChild(opt);
        });

        actionSelect.appendChild(noneGroup);
        if (movementGroup.children.length > 0) actionSelect.appendChild(movementGroup);
        if (parameterGroup.children.length > 0) actionSelect.appendChild(parameterGroup);

        const paramInput = document.createElement('input');
        paramInput.type = 'number';
        paramInput.className = 'input-field w-16 py-1 text-xs bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-600 hidden';
        paramInput.value = actionParam;
        paramInput.placeholder = '%';

        const updateStyle = () => {
            if (actionSelect.value === 'None') {
                actionSelect.classList.add('text-gray-400', 'dark:text-gray-500', 'italic');
                actionSelect.classList.remove('font-medium', 'text-gray-800', 'dark:text-white');
            } else {
                actionSelect.classList.remove('text-gray-400', 'dark:text-gray-500', 'italic');
                actionSelect.classList.add('font-medium', 'text-gray-800', 'dark:text-white');
            }
            if (actionSelect.value && actionSelect.value.startsWith('config.') && !filterSlidersOnly) {
                paramInput.classList.remove('hidden');
            } else {
                paramInput.classList.add('hidden');
            }
        };
        actionSelect.addEventListener('change', updateStyle);
        updateStyle();

        actionSelect.getFinalValue = () => {
            if (actionSelect.value && actionSelect.value.startsWith('config.') && !filterSlidersOnly) {
                return `${actionSelect.value}(${paramInput.value || 0})`;
            }
            return actionSelect.value;
        };

        fragment.appendChild(actionSelect);
        if (!filterSlidersOnly) {
            fragment.appendChild(paramInput);
        }

        return { fragment, actionSelect, paramInput };
    }

    function createSingleMappingRow(gestureList, selectedGesture, currentAction, selectClass, isStatic = false) {
        const row = document.createElement('div');
        row.className = `list-row flex flex-col sm:flex-row gap-2 py-3 items-start sm:items-center border-b border-gray-200 dark:border-gray-700`;

        const gestureDiv = document.createElement('div');
        gestureDiv.className = `flex items-center gap-2 w-full sm:w-1/2 pr-2`;
        
        const gRes = createGestureSelectEl(gestureList, selectedGesture, isStatic, `w-32 gesture-select`);
        gestureDiv.appendChild(gRes.fragment);

        if (!isStatic) {
            const infoBtn = document.createElement('button');
            infoBtn.className = 'flex-shrink-0 w-6 h-6 flex items-center justify-center rounded-full text-gray-400 hover:text-blue-500 dark:text-gray-500 dark:hover:text-blue-400 hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors ml-auto';
            infoBtn.title = `Preview example`;
            infoBtn.innerHTML = '<i class="fa-solid fa-circle-info" style="font-size:0.85em"></i>';
            infoBtn.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                showExampleVideoModal(gRes.select.value);
            });
            gestureDiv.appendChild(infoBtn);
        }

        const inputContainer = document.createElement('div');
        inputContainer.className = 'flex items-center w-full sm:w-1/2 gap-2';
        
        const aRes = createActionSelectEl(currentAction, selectClass);
        
        const removeBtn = document.createElement('button');
        removeBtn.className = 'w-8 h-8 flex-shrink-0 flex items-center justify-center rounded text-red-500 hover:bg-red-100 dark:hover:bg-red-900/30 transition-colors';
        removeBtn.innerHTML = '<i class="fa-solid fa-trash"></i>';
        removeBtn.addEventListener('click', (e) => {
            e.preventDefault();
            row.remove();
            markChanged();
        });

        inputContainer.appendChild(aRes.fragment);
        inputContainer.appendChild(removeBtn);

        row.appendChild(gestureDiv);
        row.appendChild(inputContainer);
        return row;
    }

    function createComboRow(leftGesture, rightGesture, action) {
        const id = `combo-${comboCount++}`;
        const row = document.createElement('div');
        row.className = 'flex flex-col sm:flex-row gap-2 py-3 border-b border-gray-200 dark:border-gray-700 items-start sm:items-center combo-row';
        row.id = id;

        const selectorsDiv = document.createElement('div');
        selectorsDiv.className = 'flex items-center gap-2 w-full sm:w-1/2';

        const gLeft = createGestureSelectEl(staticGesturesList, leftGesture, true, 'w-full combo-left-select', 'L: ');
        const plusSign = document.createElement('span');
        plusSign.className = 'text-gray-500 font-bold';
        plusSign.innerText = '+';
        const gRight = createGestureSelectEl(staticGesturesList, rightGesture, true, 'w-full combo-right-select', 'R: ');

        selectorsDiv.appendChild(gLeft.fragment);
        selectorsDiv.appendChild(plusSign);
        selectorsDiv.appendChild(gRight.fragment);

        const actionDiv = document.createElement('div');
        actionDiv.className = 'flex items-center gap-2 w-full sm:w-1/2';

        const aRes = createActionSelectEl(action, 'combo-action-select');

        const removeBtn = document.createElement('button');
        removeBtn.className = 'w-8 h-8 flex-shrink-0 flex items-center justify-center rounded text-red-500 hover:bg-red-100 dark:hover:bg-red-900/30 transition-colors';
        removeBtn.innerHTML = '<i class="fa-solid fa-trash"></i>';
        removeBtn.addEventListener('click', (e) => {
            e.preventDefault();
            row.remove();
            markChanged();
        });

        actionDiv.appendChild(aRes.fragment);
        actionDiv.appendChild(removeBtn);

        row.appendChild(selectorsDiv);
        row.appendChild(actionDiv);
        return row;
    }

    function createSliderRow(leftGesture, rightGesture, action) {
        const id = `slider-${sliderCount++}`;
        const row = document.createElement('div');
        row.className = 'flex flex-col sm:flex-row gap-2 py-3 border-b border-gray-200 dark:border-gray-700 items-start sm:items-center slider-row';
        row.id = id;

        const selectorsDiv = document.createElement('div');
        selectorsDiv.className = 'flex items-center gap-2 w-full sm:w-1/2';

        const gLeft = createGestureSelectEl(staticGesturesList, leftGesture, true, 'w-full slider-left-select', 'L: ');
        const plusSign = document.createElement('span');
        plusSign.className = 'text-gray-500 font-bold';
        plusSign.innerText = '+';
        const gRight = createGestureSelectEl(staticGesturesList, rightGesture, true, 'w-full slider-right-select', 'R: ');

        selectorsDiv.appendChild(gLeft.fragment);
        selectorsDiv.appendChild(plusSign);
        selectorsDiv.appendChild(gRight.fragment);

        const actionDiv = document.createElement('div');
        actionDiv.className = 'flex items-center gap-2 w-full sm:w-1/2';

        const aRes = createActionSelectEl(action, 'slider-action-select', true);

        const removeBtn = document.createElement('button');
        removeBtn.className = 'w-8 h-8 flex-shrink-0 flex items-center justify-center rounded text-red-500 hover:bg-red-100 dark:hover:bg-red-900/30 transition-colors';
        removeBtn.innerHTML = '<i class="fa-solid fa-trash"></i>';
        removeBtn.addEventListener('click', (e) => {
            e.preventDefault();
            row.remove();
            markChanged();
        });

        actionDiv.appendChild(aRes.fragment);
        actionDiv.appendChild(removeBtn);

        row.appendChild(selectorsDiv);
        row.appendChild(actionDiv);
        return row;
    }

    function createSeriesRow(seriesArray, action) {
        const id = `series-${seriesCount++}`;
        const row = document.createElement('div');
        row.className = 'flex flex-col gap-2 py-3 border-b border-gray-200 dark:border-gray-700 series-row';
        row.id = id;

        const stepsContainer = document.createElement('div');
        stepsContainer.className = 'flex items-center gap-2 w-full series-steps-container';
        stepsContainer.style.flexWrap = 'wrap';

        const addStepBtn = document.createElement('button');
        addStepBtn.className = 'text-xs text-blue-500 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300 p-1 rounded font-bold';
        addStepBtn.innerText = '+ Add Step';

        function createStepSelect(val) {
            const stepDiv = document.createElement('div');
            stepDiv.className = 'flex items-center gap-1 series-step w-auto';
            stepDiv.style.flexShrink = '0';
            stepDiv.style.marginBottom = '8px';

            const handSelect = document.createElement('select');
            handSelect.className = 'input-field text-xs py-1 px-1 series-hand-select bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-600 !w-12 shrink-0';
            ['A', 'L', 'R'].forEach(h => {
                const opt = document.createElement('option');
                opt.value = h;
                opt.innerText = h;
                if (val && val.startsWith(`${h}:`)) opt.selected = true;
                handSelect.appendChild(opt);
            });

            let selectedG = 'Closed_Fist';
            if (val) {
                const parts = val.split(':');
                if (parts.length === 2) selectedG = parts[1];
            }
            const gRes = createGestureSelectEl(staticGesturesList, selectedG, true, 'px-1 series-gesture-select shrink-0 !w-auto');

            const removeStepBtn = document.createElement('button');
            removeStepBtn.className = 'text-red-500 px-1 font-bold';
            removeStepBtn.innerHTML = '&times;';
            removeStepBtn.addEventListener('click', (e) => {
                e.preventDefault();
                if (stepsContainer.querySelectorAll('.series-step').length > 1) {
                    stepDiv.remove();
                    renderArrows();
                    markChanged();
                }
            });

            stepDiv.appendChild(handSelect);
            stepDiv.appendChild(gRes.fragment);
            stepDiv.appendChild(removeStepBtn);

            const arrow = document.createElement('span');
            arrow.className = 'text-gray-500 font-bold step-arrow ml-1';
            arrow.innerHTML = '<i class="fa-solid fa-arrow-right"></i>';
            stepDiv.appendChild(arrow);

            return stepDiv;
        }

        const renderArrows = () => {
            const steps = stepsContainer.querySelectorAll('.series-step');
            steps.forEach((step, idx) => {
                const arrow = step.querySelector('.step-arrow');
                if (idx === steps.length - 1) arrow.style.display = 'none';
                else arrow.style.display = 'inline-block';
            });
        };

        if (!seriesArray || seriesArray.length === 0) seriesArray = ["A:Closed_Fist"];

        seriesArray.forEach(val => {
            stepsContainer.appendChild(createStepSelect(val));
        });

        stepsContainer.appendChild(addStepBtn);
        renderArrows();

        addStepBtn.addEventListener('click', (e) => {
            e.preventDefault();
            stepsContainer.insertBefore(createStepSelect('A:Open_Palm'), addStepBtn);
            renderArrows();
            markChanged();
        });

        const actionDiv = document.createElement('div');
        actionDiv.className = 'flex items-center gap-2 w-full mt-2';
        actionDiv.style.marginTop = '8px';

        const aRes = createActionSelectEl(action, 'series-action-select');

        const removeBtn = document.createElement('button');
        removeBtn.className = 'w-8 h-8 flex-shrink-0 flex items-center justify-center rounded text-red-500 hover:bg-red-100 dark:hover:bg-red-900/30 transition-colors';
        removeBtn.innerHTML = '<i class="fa-solid fa-trash"></i>';
        removeBtn.addEventListener('click', (e) => {
            e.preventDefault();
            row.remove();
            markChanged();
        });

        actionDiv.appendChild(aRes.fragment);
        actionDiv.appendChild(removeBtn);

        row.appendChild(stepsContainer);
        row.appendChild(actionDiv);

        return row;
    }

    const handleThemeToggle = () => {
        const isDark = document.documentElement.classList.toggle('dark');
        localStorage.setItem('theme', isDark ? 'dark' : 'light');
    };

    const handleSimModeToggle = async () => {
        if (currentHasControl === null) return;

        toggleSimModeBtn.disabled = true;
        const previousContent = toggleSimModeBtn.innerHTML;
        toggleSimModeBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i>';

        const wantControl = !currentHasControl;
        const modeName = wantControl ? 'Live' : 'Simulation';

        // Create log modal dynamically
        const modalId = 'sim-log-modal';
        let modal = document.getElementById(modalId);
        if (!modal) {
            modal = document.createElement('div');
            modal.id = modalId;
            modal.className = 'fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-75';
            modal.innerHTML = `
                <div class="bg-gray-900 border border-gray-700 rounded-lg shadow-xl w-11/12 max-w-2xl flex flex-col h-[60vh] max-h-96">
                    <div class="flex justify-between items-center p-3 border-b border-gray-700 bg-gray-800 rounded-t-lg">
                        <h3 class="text-gray-200 font-semibold font-mono text-sm">Switching to ${modeName} Mode...</h3>
                        <button id="${modalId}-close" class="text-gray-400 hover:text-white transition-colors disabled:opacity-50" disabled>
                            <i class="fa-solid fa-xmark"></i>
                        </button>
                    </div>
                    <div id="${modalId}-logs" class="flex-1 p-4 overflow-y-auto font-mono text-xs text-green-400 bg-black whitespace-pre-wrap flex flex-col gap-1"></div>
                </div>
            `;
            document.body.appendChild(modal);
        } else {
            modal.querySelector('h3').innerText = `Switching to ${modeName} Mode...`;
            modal.querySelector(`#${modalId}-logs`).innerHTML = '';
            modal.querySelector(`#${modalId}-close`).disabled = true;
            modal.classList.remove('hidden');
        }

        const logContainer = modal.querySelector(`#${modalId}-logs`);
        const closeBtn = modal.querySelector(`#${modalId}-close`);
        
        closeBtn.onclick = () => {
            modal.classList.add('hidden');
        };

        const appendLog = (msg, isError = false) => {
            const el = document.createElement('div');
            el.className = isError ? 'text-red-400' : 'text-green-400';
            el.textContent = msg;
            logContainer.appendChild(el);
            logContainer.scrollTop = logContainer.scrollHeight;
        };

        try {
            const response = await updateHasControl(wantControl);
            
            if (!response.ok) {
                const data = await response.json().catch(() => ({}));
                throw new Error(data.error || 'Failed to switch mode (HTTP ' + response.status + ')');
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop(); // keep remainder
                
                for (const line of lines) {
                    if (!line.trim()) continue;
                    try {
                        const data = JSON.parse(line);
                        if (data.log) {
                            appendLog(data.log);
                        } else if (data.error) {
                            throw new Error(data.error);
                        } else if ('has_control' in data) {
                            // Success!
                            currentHasControl = data.has_control;
                            const statusText = !data.has_control ? '🔴 Simulation Mode' : '🟢 Live Mode';
                            const statusColor = !data.has_control ? 'text-yellow-600 dark:text-yellow-400' : 'text-green-600 dark:text-green-400';
                            simModeStatus.innerHTML = `<span class="${statusColor} font-semibold">${statusText}</span>`;
                            markChanged();
                            appendLog(`\n✓ Successfully switched to ${modeName} Mode.`);
                            toggleSimModeBtn.innerHTML = `✓ Switched to ${modeName}`;
                        }
                    } catch (err) {
                        if (err.message !== "Unexpected end of JSON input" && !err.message.includes("JSON")) {
                            throw err; // Re-throw non JSON parsing errors
                        }
                    }
                }
            }

            setTimeout(() => {
                toggleSimModeBtn.innerHTML = previousContent;
                toggleSimModeBtn.disabled = false;
            }, 2000);
            
            closeBtn.disabled = false;

        } catch (e) {
            console.error('Failed to toggle simulation mode:', e);
            appendLog(`\n✗ Error: ${e.message}`, true);
            simModeStatus.innerHTML = `<span class="text-red-600 dark:text-red-400 font-semibold">⚠ Switch Failed</span>`;
            toggleSimModeBtn.innerHTML = `✗ Failed`;

            setTimeout(() => {
                toggleSimModeBtn.innerHTML = previousContent;
                toggleSimModeBtn.disabled = false;
            }, 3000);
            
            closeBtn.disabled = false;
        }
    };

    const updateSimModeStatus = (hasControl) => {
        if (simModeStatus) {
            const statusText = !hasControl ? '🔴 Simulation Mode' : '🟢 Live Mode';
            const statusColor = !hasControl ? 'text-yellow-600 dark:text-yellow-400' : 'text-green-600 dark:text-green-400';
            simModeStatus.innerHTML = `<span class="${statusColor} font-semibold">${statusText}</span>`;
        }
    };

    const hasControlChangeListener = (e) => {
        currentHasControl = e.detail.hasControl;
        updateSimModeStatus(currentHasControl);
    };
    window.addEventListener('hasControlChanged', hasControlChangeListener);

    const updateToggleUI = (btn, isEnabled) => {
        const thumb = btn.querySelector('span');
        if (isEnabled) {
            btn.classList.remove('bg-gray-300', 'dark:bg-gray-600');
            btn.classList.add('bg-blue-500', 'dark:bg-blue-600');
            thumb.classList.remove('translate-x-1');
            thumb.classList.add('translate-x-5');
        } else {
            btn.classList.add('bg-gray-300', 'dark:bg-gray-600');
            btn.classList.remove('bg-blue-500', 'dark:bg-blue-600');
            thumb.classList.add('translate-x-1');
            thumb.classList.remove('translate-x-5');
        }
    };

    container.querySelector('#show-unmapped-toggle').addEventListener('click', (e) => {
        e.preventDefault();
        const btn = e.currentTarget;
        const isEnabled = btn.classList.contains('bg-blue-500');
        updateToggleUI(btn, !isEnabled);
        markChanged();
    });

    container.querySelector('#dynamic-conf-slider').addEventListener('input', (e) => {
        container.querySelector('#dynamic-conf-val').innerText = parseFloat(e.target.value).toFixed(2);
    });

    staticConfSlider.addEventListener('input', (e) => {
        staticConfVal.innerText = parseFloat(e.target.value).toFixed(2);
    });

    twoHandWaitSlider.addEventListener('input', (e) => {
        twoHandWaitVal.innerText = e.target.value;
    });

    if (seriesTimeoutSlider) {
        seriesTimeoutSlider.addEventListener('input', (e) => {
            if (seriesTimeoutVal) {
                seriesTimeoutVal.innerText = e.target.value;
            }
        });
    }

    addComboBtn.addEventListener('click', (e) => {
        e.preventDefault();
        twoHandContainer.appendChild(createComboRow('Closed_Fist', 'Open_Palm', 'None'));
        markChanged();
    });

    if (resetStaticBtn) {
        resetStaticBtn.addEventListener('click', (e) => {
            e.preventDefault();
            staticLeftContainer.innerHTML = '';
            staticRightContainer.innerHTML = '';
            markChanged();
        });
    }

    if (resetDynamicBtn) {
        resetDynamicBtn.addEventListener('click', (e) => {
            e.preventDefault();
            const mappingsContainer = container.querySelector('#gesture-mappings-container');
            if (mappingsContainer) mappingsContainer.innerHTML = '';
            markChanged();
        });
    }

    if (resetComboBtn) {
        resetComboBtn.addEventListener('click', (e) => {
            e.preventDefault();
            twoHandContainer.innerHTML = '';
            markChanged();
        });
    }

    if (resetSliderBtn) {
        resetSliderBtn.addEventListener('click', (e) => {
            e.preventDefault();
            sliderContainer.innerHTML = '';
            markChanged();
        });
    }

    if (resetSeriesBtn) {
        resetSeriesBtn.addEventListener('click', (e) => {
            e.preventDefault();
            seriesContainer.innerHTML = '';
            markChanged();
        });
    }

    addSliderBtn.addEventListener('click', (e) => {
        e.preventDefault();
        sliderContainer.appendChild(createSliderRow('Closed_Fist', 'Open_Palm', 'None'));
        markChanged();
    });

    addSeriesBtn.addEventListener('click', (e) => {
        e.preventDefault();
        seriesContainer.appendChild(createSeriesRow(['A:Closed_Fist', 'A:Open_Palm'], 'None'));
        markChanged();
    });

    const handleMainAction = async () => {
        if (!hasUnsavedChanges) {
            onClose();
            return;
        }

        mainBtn.innerText = 'Saving...';
        mainBtn.disabled = true;
        try {
            const spotPromise = updateSpotConfig({
                VELOCITY_BASE_SPEED: parseFloat(container.querySelector('#spot-vel-speed').value) || 1.0,
                VELOCITY_BASE_ANGULAR: parseFloat(container.querySelector('#spot-vel-angular').value) || 0.8,
                VELOCITY_CMD_DURATION: parseFloat(container.querySelector('#spot-cmd-duration').value) || 1.0,
                BODY_PITCH_ANGLE: parseFloat(container.querySelector('#spot-pitch-angle').value) || 0.4,
                BODY_YAW_ANGLE: parseFloat(container.querySelector('#spot-yaw-angle').value) || 0.4
            });

            // Gather dynamic gestures
            const newDynamicMappings = {};
            globalDynamicGesturesList.forEach(g => newDynamicMappings[g] = null);
            container.querySelectorAll('.dynamic-mapping-select').forEach(select => {
                const val = select.getFinalValue ? select.getFinalValue() : select.value;
                const row = select.closest('.list-row');
                const gesture = row.querySelector('.gesture-select').value;
                newDynamicMappings[gesture] = val === 'None' ? null : val;
            });
            const gesturePromise = updateGestureConfig({
                mappings: newDynamicMappings,
                confidence_threshold: parseFloat(container.querySelector('#dynamic-conf-slider').value),
                show_unmapped: container.querySelector('#show-unmapped-toggle').classList.contains('bg-blue-500')
            });

            // Gather static gestures
            const newLeftMappings = {};
            staticGesturesList.forEach(g => newLeftMappings[g] = null);
            container.querySelectorAll('.static-left-select').forEach(select => {
                const val = select.getFinalValue ? select.getFinalValue() : select.value;
                const row = select.closest('.list-row');
                const gesture = row.querySelector('.gesture-select').value;
                newLeftMappings[gesture] = val === 'None' ? null : val;
            });

            const newRightMappings = {};
            staticGesturesList.forEach(g => newRightMappings[g] = null);
            container.querySelectorAll('.static-right-select').forEach(select => {
                const val = select.getFinalValue ? select.getFinalValue() : select.value;
                const row = select.closest('.list-row');
                const gesture = row.querySelector('.gesture-select').value;
                newRightMappings[gesture] = val === 'None' ? null : val;
            });
            const staticPromise = updateStaticGestureConfig({
                mappings_left: newLeftMappings,
                mappings_right: newRightMappings,
                confidence_threshold: parseFloat(staticConfSlider.value)
            });

            // Gather two hand combos and sliders
            const newCombos = {};
            container.querySelectorAll('.combo-row').forEach(row => {
                const left = row.querySelector('.combo-left-select').value;
                const right = row.querySelector('.combo-right-select').value;
                const select = row.querySelector('.combo-action-select');
                const action = select.getFinalValue ? select.getFinalValue() : select.value;
                if (action !== 'None') {
                    newCombos[`${left}+${right}`] = action;
                }
            });
            container.querySelectorAll('.slider-row').forEach(row => {
                const left = row.querySelector('.slider-left-select').value;
                const right = row.querySelector('.slider-right-select').value;
                const select = row.querySelector('.slider-action-select');
                const action = select.getFinalValue ? select.getFinalValue() : select.value;
                if (action !== 'None') {
                    newCombos[`${left}+${right}`] = action;
                }
            });

            const newSeries = {};
            container.querySelectorAll('.series-row').forEach(row => {
                const steps = Array.from(row.querySelectorAll('.series-step')).map(step => {
                    const hand = step.querySelector('.series-hand-select').value;
                    const gesture = step.querySelector('.series-gesture-select').value;
                    return `${hand}:${gesture}`;
                });

                const select = row.querySelector('.series-action-select');
                const action = select.getFinalValue ? select.getFinalValue() : select.value;
                if (action !== 'None' && steps.length > 0) {
                    newSeries[steps.join(',')] = action;
                }
            });

            const twoHandPromise = updateTwoHandConfig({
                two_hand_wait_ms: parseInt(twoHandWaitSlider.value),
                series_timeout_ms: parseInt(seriesTimeoutSlider.value),
                combinations: newCombos,
                series: newSeries
            });

            await Promise.all([spotPromise, gesturePromise, staticPromise, twoHandPromise]);
            onClose();
        } catch (e) {
            console.error("Failed to save settings", e);
            mainBtn.innerText = 'Save & Close';
            mainBtn.disabled = false;
        }
    };

    const isManualEnabled = localStorage.getItem('manualControlsEnabled') !== 'false';
    updateToggleUI(manualControlToggleBtn, isManualEnabled);

    const handleManualControlToggle = () => {
        const currentlyEnabled = localStorage.getItem('manualControlsEnabled') !== 'false';
        const nextState = !currentlyEnabled;
        localStorage.setItem('manualControlsEnabled', nextState ? 'true' : 'false');
        updateToggleUI(manualControlToggleBtn, nextState);
    };

    manualControlToggleBtn.addEventListener('click', handleManualControlToggle);
    themeToggleBtn.addEventListener('click', handleThemeToggle);
    toggleSimModeBtn.addEventListener('click', handleSimModeToggle);
    closeBtn.addEventListener('click', onClose);
    mainBtn.addEventListener('click', handleMainAction);

    resetDynamicBtn.addEventListener('click', (e) => {
        e.preventDefault();
        container.querySelectorAll('.dynamic-mapping-select').forEach(select => {
            select.value = 'None';
            select.dispatchEvent(new Event('change'));
        });
        markChanged();
    });

    resetStaticBtn.addEventListener('click', (e) => {
        e.preventDefault();
        container.querySelectorAll('.static-left-select, .static-right-select').forEach(select => {
            select.value = 'None';
            select.dispatchEvent(new Event('change'));
        });
        markChanged();
    });

    resetComboBtn.addEventListener('click', (e) => {
        e.preventDefault();
        container.querySelectorAll('.combo-action-select').forEach(select => {
            select.value = 'None';
            select.dispatchEvent(new Event('change'));
        });
        markChanged();
    });

    resetSliderBtn.addEventListener('click', (e) => {
        e.preventDefault();
        container.querySelectorAll('.slider-action-select').forEach(select => {
            select.value = 'None';
            select.dispatchEvent(new Event('change'));
        });
        markChanged();
    });

    resetSeriesBtn.addEventListener('click', (e) => {
        e.preventDefault();
        container.querySelectorAll('.series-action-select').forEach(select => {
            select.value = 'None';
            select.dispatchEvent(new Event('change'));
        });
        markChanged();
    });

    // ── Example video modal ──────────────────────────────────────────────
    function showExampleVideoModal(gestureName) {
        // Build overlay
        const overlay = document.createElement('div');
        overlay.className = 'fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-60';
        overlay.style.backdropFilter = 'blur(4px)';

        // Modal card
        const card = document.createElement('div');
        card.className = 'relative bg-white dark:bg-gray-900 rounded-xl shadow-2xl overflow-hidden max-w-lg w-full mx-4';

        // Header
        const header = document.createElement('div');
        header.className = 'flex items-center justify-between px-5 py-3 border-b dark:border-gray-700';

        const title = document.createElement('h3');
        title.className = 'text-lg font-bold text-gray-800 dark:text-white truncate';
        title.innerText = gestureName;

        const closeBtn = document.createElement('button');
        closeBtn.className = 'w-8 h-8 flex items-center justify-center rounded-full text-gray-400 hover:text-gray-700 dark:hover:text-white hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors';
        closeBtn.innerHTML = '<i class="fa-solid fa-xmark"></i>';

        header.appendChild(title);
        header.appendChild(closeBtn);

        // Video container
        const videoContainer = document.createElement('div');
        videoContainer.className = 'bg-black flex items-center justify-center';
        videoContainer.style.minHeight = '200px';

        // Spinner while loading
        const spinner = document.createElement('div');
        spinner.className = 'text-white text-center py-8';
        spinner.innerHTML = '<i class="fa-solid fa-spinner fa-spin fa-2x"></i><p class="mt-3 text-sm text-gray-400">Loading example...</p>';
        videoContainer.appendChild(spinner);

        // Video element
        const video = document.createElement('video');
        video.className = 'w-full hidden';
        video.setAttribute('autoplay', '');
        video.setAttribute('muted', '');
        video.setAttribute('playsinline', '');

        // Build the video URL from the gesture name
        const encodedName = encodeURIComponent(gestureName);
        video.src = `${BACKEND_URL}/example_video/${encodedName}`;

        video.addEventListener('loadeddata', () => {
            spinner.classList.add('hidden');
            video.classList.remove('hidden');
        });

        video.addEventListener('error', () => {
            spinner.innerHTML = '<p class="text-red-400 text-sm">No example video available</p>';
        });

        // Auto-close when video ends
        video.addEventListener('ended', () => {
            cleanup();
        });

        videoContainer.appendChild(video);
        card.appendChild(header);
        card.appendChild(videoContainer);
        overlay.appendChild(card);
        document.body.appendChild(overlay);

        // Cleanup function
        function cleanup() {
            video.pause();
            video.removeAttribute('src');
            video.load(); // release resources
            if (overlay.parentNode) {
                overlay.parentNode.removeChild(overlay);
            }
        }

        // Close handlers
        closeBtn.addEventListener('click', cleanup);
        overlay.addEventListener('click', (e) => {
            if (e.target === overlay) cleanup();
        });
    }

    return () => {
        formBody.removeEventListener('input', markChanged);
        formBody.removeEventListener('change', markChanged);
        themeToggleBtn.removeEventListener('click', handleThemeToggle);
        manualControlToggleBtn.removeEventListener('click', handleManualControlToggle);
        toggleSimModeBtn.removeEventListener('click', handleSimModeToggle);
        closeBtn.removeEventListener('click', onClose);
        mainBtn.removeEventListener('click', handleMainAction);
        window.removeEventListener('hasControlChanged', hasControlChangeListener);
    };
}
