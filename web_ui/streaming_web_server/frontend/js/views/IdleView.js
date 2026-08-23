import { startStream } from '../api.js';

export function renderIdleView(container, navigateTo) {
    container.innerHTML = `
        <div class="absolute top-4 right-4 sm:top-8 sm:right-8">
            <button id="settings-btn" class="btn w-12 h-12 bg-white dark:bg-gray-800 !border-gray-300 dark:!border-gray-600 border text-gray-700 dark:text-gray-200 hover:bg-gray-50 dark:hover:bg-gray-700">
                <i class="fa-solid fa-gear text-xl"></i>
            </button>
        </div>
        <h1 class="text-4xl font-bold text-gray-800 dark:text-white mt-8 tracking-tight text-center">Robot Gesture Control</h1>
        <div class="h-full w-full flex flex-col items-center justify-center p-4">
            <button id="start-btn" class="btn px-10 py-5 bg-green-600 text-white text-2xl shadow-lg hover:shadow-xl hover:bg-green-700 dark:bg-green-700 dark:hover:bg-green-600">
                <i class="fa-solid fa-play mr-3"></i>
                Start
            </button>
        </div>
    `;

    const startBtn = container.querySelector('#start-btn');
    const settingsBtn = container.querySelector('#settings-btn');

    let isTransitioning = false;

    const handleStart = async () => {
        if (isTransitioning) return;
        isTransitioning = true;
        startBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Starting...';
        startBtn.classList.add('opacity-80', 'cursor-not-allowed');

        try {
            await startStream();
            // App state loop will pick up "starting" state automatically
            // But we can trigger immediate visual feedback by updating button text
        } catch (e) {
            console.error("Failed to start stream:", e);
            startBtn.innerHTML = '<i class="fa-solid fa-play"></i> Start Stream';
            startBtn.classList.remove('opacity-80', 'cursor-not-allowed');
            isTransitioning = false;
        }
    };

    const handleSettings = () => {
        navigateTo('@settings');
    };

    startBtn.addEventListener('click', handleStart);
    settingsBtn.addEventListener('click', handleSettings);

    return () => {
        startBtn.removeEventListener('click', handleStart);
        settingsBtn.removeEventListener('click', handleSettings);
    };
}
