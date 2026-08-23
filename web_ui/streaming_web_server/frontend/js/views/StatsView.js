import { fetchStatsReport } from '../api.js';

export function renderStatsView(container, onClose) {
    container.innerHTML = `
        <div class="flex flex-col h-full bg-gray-100 dark:bg-gray-900 overflow-hidden relative transition-colors duration-200">
            <!-- Header bar fixed top -->
            <div class="flex-shrink-0 bg-white dark:bg-gray-800 shadow z-10 px-6 py-4 flex justify-between items-center transition-colors duration-200">
                <h2 class="text-2xl font-bold text-gray-800 dark:text-gray-100">Run Statistics</h2>
                <button id="close-stats-btn" class="btn bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 text-gray-800 dark:text-gray-200 px-4 py-2">
                    <i class="fa-solid fa-xmark"></i> Close
                </button>
            </div>
            
            <!-- Scrollable content list of images -->
            <div class="flex-1 overflow-y-auto p-4 md:p-8" id="stats-scroll-area">
                <div id="stats-loading" class="text-center py-10 text-gray-500 dark:text-gray-400">
                    <i class="fa-solid fa-spinner fa-spin fa-2x mb-4"></i>
                    <p>Loading stat charts...</p>
                </div>
                <div id="stats-grid" class="flex flex-col gap-8 max-w-4xl mx-auto w-full items-center">
                    <!-- Images will be inserted here dynamically -->
                </div>
            </div>
        </div>
    `;

    const closeBtn = container.querySelector('#close-stats-btn');
    const scrollArea = container.querySelector('#stats-scroll-area');
    const loadingDiv = container.querySelector('#stats-loading');
    const gridDiv = container.querySelector('#stats-grid');

    const handleClose = () => onClose();
    closeBtn.addEventListener('click', handleClose);

    let isFetching = true;
    let autoScrollInterval = null;
    let autoScrollTimeout = null;

    // Load Charts
    fetchStatsReport()
        .then(images => {
            isFetching = false;
            loadingDiv.style.display = 'none';

            if (!images || images.length === 0) {
                gridDiv.innerHTML = '<p class="text-gray-500 italic mt-8">No stats were generated for this run.</p>';
                return;
            }

            images.forEach(item => {
                const img = document.createElement('img');
                img.src = item.data;
                img.alt = item.filename;
                img.title = item.filename;
                img.className = "w-full border shadow rounded-lg bg-white object-contain";
                gridDiv.appendChild(img);
            });

            // Start auto scroll
            startAutoScroll();

        })
        .catch(err => {
            console.error('Error fetching stats report:', err);
            isFetching = false;
            loadingDiv.innerHTML = '<p class="text-red-500">Failed to load statistics.</p>';
        });

    // Auto-scroll logic preserving previous user-requested feature
    const AUTO_SCROLL_DELAY = 5000;
    const SCROLL_SPEED = 2;
    let scrollDirection = 1;

    const scrollStep = () => {
        if (!scrollArea) return;
        const scrollHeight = scrollArea.scrollHeight - scrollArea.clientHeight;
        scrollArea.scrollTop += SCROLL_SPEED * scrollDirection;
        if (scrollDirection === 1 && scrollArea.scrollTop >= scrollHeight) {
            scrollDirection = -1;
            pauseAndResume();
        } else if (scrollDirection === -1 && scrollArea.scrollTop <= 0) {
            scrollDirection = 1;
            pauseAndResume();
        }
    };

    const pauseAndResume = () => {
        clearInterval(autoScrollInterval);
        autoScrollInterval = null;
        autoScrollTimeout = setTimeout(() => {
            autoScrollInterval = setInterval(scrollStep, 30);
        }, AUTO_SCROLL_DELAY);
    };

    const startAutoScroll = () => {
        stopAutoScroll();
        scrollDirection = 1;
        autoScrollTimeout = setTimeout(() => {
            if (scrollArea.scrollHeight > scrollArea.clientHeight) {
                autoScrollInterval = setInterval(scrollStep, 30);
            }
        }, AUTO_SCROLL_DELAY);
    };

    const stopAutoScroll = () => {
        if (autoScrollTimeout) clearTimeout(autoScrollTimeout);
        if (autoScrollInterval) clearInterval(autoScrollInterval);
        autoScrollTimeout = null;
        autoScrollInterval = null;
    };

    // Listen to manual scrolling
    const onManualScroll = () => {
        stopAutoScroll();
        setTimeout(startAutoScroll, AUTO_SCROLL_DELAY); // resume eventually
    };
    scrollArea.addEventListener('wheel', onManualScroll);
    scrollArea.addEventListener('touchstart', onManualScroll);

    return () => {
        closeBtn.removeEventListener('click', handleClose);
        scrollArea.removeEventListener('wheel', onManualScroll);
        scrollArea.removeEventListener('touchstart', onManualScroll);
        stopAutoScroll();
    };
}
