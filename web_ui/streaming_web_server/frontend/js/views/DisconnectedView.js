export function renderDisconnectedView(container) {
    container.innerHTML = `
        <div class="flex-1 flex flex-col items-center justify-center p-4 text-center bg-gray-100 dark:bg-gray-900 transition-colors duration-200 h-full w-full">
            <div class="text-red-500 dark:text-red-400 mb-6">
                <i class="fa-solid fa-circle-xmark fa-5x"></i>
            </div>
            <h1 class="text-3xl font-bold text-gray-800 dark:text-gray-100 mb-2">Backend Unreachable</h1>
            <p class="text-gray-600 dark:text-gray-400 max-w-md">
                We couldn't connect to the backend server. Please make sure the gesture backend stream is running and accessible at 
                <strong class="text-gray-800 dark:text-gray-200">${window.BACKEND_URL}</strong>.
            </p>
            <button id="retry-btn" class="btn mt-8 px-6 py-2 bg-blue-600 text-white hover:bg-blue-700 dark:bg-blue-700 dark:hover:bg-blue-600">
                Retry Connection
            </button>
        </div>
    `;

    const retryBtn = container.querySelector('#retry-btn');
    const handleRetry = () => window.location.reload();
    retryBtn.addEventListener('click', handleRetry);

    return () => {
        retryBtn.removeEventListener('click', handleRetry);
    };
}
