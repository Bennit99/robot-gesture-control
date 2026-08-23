export function renderLoadingView(container, reason) {
    const text = reason === 'stopping'
        ? 'Stopping stream and generating stats...'
        : 'Preparing stream...';

    container.innerHTML = `
        <div class="h-full w-full flex flex-col items-center justify-center p-4 text-center m-auto bg-gray-100 dark:bg-gray-900 transition-colors duration-200">
            <div class="text-blue-500 mb-6">
                <i class="fa-solid fa-circle-notch fa-spin fa-4x"></i>
            </div>
            <h2 class="text-2xl font-bold text-gray-800 dark:text-gray-100 mb-2">${text}</h2>
            <p class="text-gray-500 dark:text-gray-400">Please wait. This might take a moment...</p>
        </div>
    `;

    return () => {
        // No heavy listeners to cleanup
    };
}
