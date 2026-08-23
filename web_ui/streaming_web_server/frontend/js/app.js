import { fetchStreamState, fetchHasControl, triggerEstop, triggerManualControl } from './api.js';
import { renderDisconnectedView } from './views/DisconnectedView.js';
import { renderIdleView } from './views/IdleView.js';
import { renderRecognitionView } from './views/RecognitionView.js';
import { renderLoadingView } from './views/LoadingView.js';
import { renderStatsView } from './views/StatsView.js';
import { renderSettingsView } from './views/SettingsView.js';

const appDiv = document.getElementById('app');
const globalControls = document.createElement('div');
globalControls.id = 'global-controls';
globalControls.className = 'fixed inset-0 pointer-events-none z-50';
document.body.appendChild(globalControls);

let currentState = null;
let currentViewCleanup = null;
let currentGlobalControlsCleanup = null;
let statsDismissed = false;
let estopInFlight = false;
let currentHasControl = false;

function isTypingTarget(target) {
    if (!(target instanceof HTMLElement)) {
        return false;
    }
    return target.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName);
}

function renderGlobalControls() {
    if (currentGlobalControlsCleanup) {
        currentGlobalControlsCleanup();
        currentGlobalControlsCleanup = null;
    }

    globalControls.innerHTML = '';

    if (!currentHasControl) {
        return;
    }

    globalControls.innerHTML = `
        <div class="absolute top-4 left-4 sm:top-8 sm:left-8 pointer-events-auto">
            <button id="estop-btn" class="btn px-5 py-3 bg-red-600 text-white font-extrabold shadow-lg shadow-red-950/20 border-red-300/30 hover:bg-red-700 focus:ring-4 focus:ring-red-300/40">
                <i class="fa-solid fa-triangle-exclamation"></i>
                E-Stop
                <span class="hidden sm:inline text-xs font-semibold uppercase tracking-[0.2em] opacity-80">Space</span>
            </button>
        </div>
    `;

    const estopBtn = globalControls.querySelector('#estop-btn');

    const handleEstop = async () => {
        if (estopInFlight) return;
        estopInFlight = true;
        const previousContent = estopBtn.innerHTML;
        estopBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Engaging...';
        estopBtn.classList.add('opacity-80', 'cursor-not-allowed');
        estopBtn.disabled = true;

        try {
            await triggerEstop();
            estopBtn.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i> E-Stop Sent';
        } catch (error) {
            console.error('[App] Failed to trigger E-Stop:', error);
            estopBtn.innerHTML = previousContent;
            estopBtn.classList.remove('opacity-80', 'cursor-not-allowed');
            estopBtn.disabled = false;
            estopInFlight = false;
        }
    };

    let lastManualCmdTime = 0;
    const activeManualKeys = new Set();

    const handleKeydown = (event) => {
        if (estopInFlight || isTypingTarget(event.target)) {
            return;
        }
        if (event.code === 'Space') {
            event.preventDefault();
            if (!event.repeat) handleEstop();
            return;
        }

        const validKeys = ['w', 'a', 's', 'd', 'q', 'e'];
        const key = event.key.toLowerCase();
        
        if (validKeys.includes(key)) {
            // Check if manual controls are disabled in settings
            if (localStorage.getItem('manualControlsEnabled') === 'false') {
                return;
            }

            activeManualKeys.add(key);
            const now = Date.now();
            // Throttle network requests to max 5 per second
            if (now - lastManualCmdTime > 200) {
                lastManualCmdTime = now;
                triggerManualControl(key).catch(err => console.error('[App] Manual control failed:', err));
            }
        }
    };

    const handleKeyup = (event) => {
        if (isTypingTarget(event.target)) return;
        const key = event.key.toLowerCase();
        if (activeManualKeys.has(key)) {
            activeManualKeys.delete(key);
            // If no more manual keys are pressed, stop
            if (activeManualKeys.size === 0) {
                triggerManualControl(' ').catch(err => console.error('[App] Stop command failed:', err));
            }
        }
    };

    estopBtn.addEventListener('click', handleEstop);
    document.addEventListener('keydown', handleKeydown);
    document.addEventListener('keyup', handleKeyup);

    currentGlobalControlsCleanup = () => {
        estopBtn.removeEventListener('click', handleEstop);
        document.removeEventListener('keydown', handleKeydown);
        document.removeEventListener('keyup', handleKeyup);
    };
}

async function syncHasControl() {
    const hasControlData = await fetchHasControl();
    if (!hasControlData || typeof hasControlData.has_control !== 'boolean') {
        return;
    }

    if (hasControlData.has_control !== currentHasControl) {
        currentHasControl = hasControlData.has_control;
        renderGlobalControls();
        window.dispatchEvent(new CustomEvent('hasControlChanged', { detail: { hasControl: currentHasControl } }));
    }
}

// Allow views to request manual navigation (like going to settings)
// A custom state prefixed with "@" handles frontend-only states
export function navigateTo(frontendState) {
    if (currentState === '@stats' && frontendState === 'idle') {
        statsDismissed = true;
    }
    applyState(frontendState);
}

async function pollState() {
    await syncHasControl();

    // Suspend backend polling overrides if we are on a frontend-exclusive page like settings or stats
    if (currentState === '@settings' || currentState === '@stats') return;

    const nextState = await fetchStreamState();

    if (nextState === 'starting' || nextState === 'running') {
        statsDismissed = false;
    }

    let computedState = nextState;
    if (nextState === 'ready') {
        computedState = statsDismissed ? 'idle' : '@stats';
    }

    if (computedState !== currentState) {
        applyState(computedState);
    }
}

function applyState(nextState) {
    if (currentState === nextState) return;

    // Cleanup previous view logic (e.g. stop event loops, fetch requests, UI listeners)
    if (currentViewCleanup && typeof currentViewCleanup === 'function') {
        currentViewCleanup();
    }

    currentState = nextState;
    console.log('[App] Transitioning to state:', nextState);

    appDiv.innerHTML = '';

    switch (nextState) {
        case 'disconnected':
            currentViewCleanup = renderDisconnectedView(appDiv);
            break;
        case 'idle':
            currentViewCleanup = renderIdleView(appDiv, navigateTo);
            break;
        case 'starting':
        case 'stopping':
            currentViewCleanup = renderLoadingView(appDiv, nextState);
            break;
        case 'running':
            currentViewCleanup = renderRecognitionView(appDiv, navigateTo);
            break;
        case '@stats':
            currentViewCleanup = renderStatsView(appDiv, () => navigateTo('idle'));
            break;
        case '@settings':
            currentViewCleanup = renderSettingsView(appDiv, () => navigateTo('idle'));
            break;
        default:
            currentViewCleanup = renderIdleView(appDiv, navigateTo);
            break;
    }
}

// Initial fetch and poll
syncHasControl().then(() => {
    renderGlobalControls();
    pollState();
});
setInterval(pollState, 1500);
