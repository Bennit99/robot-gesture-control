export const BACKEND_URL = window.BACKEND_URL;

export async function fetchStreamState() {
    try {
        const response = await fetch(`${BACKEND_URL}/stream_state`, { cache: 'no-store' });
        if (!response.ok) {
            return 'disconnected';
        }
        const payload = await response.json();
        return payload.state || 'idle';
    } catch (err) {
        return 'disconnected';
    }
}

export async function startStream() {
    return fetch(`${BACKEND_URL}/start`, { method: 'POST' });
}

export async function stopStream() {
    return fetch(`${BACKEND_URL}/stop`, { method: 'POST' });
}

export async function triggerEstop() {
    const response = await fetch(`${BACKEND_URL}/estop`, { method: 'POST' });
    if (!response.ok) {
        throw new Error(`E-Stop request failed with status ${response.status}`);
    }
    return response;
}

export async function triggerManualControl(action) {
    const response = await fetch(`${BACKEND_URL}/manual_control`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action })
    });
    if (!response.ok) {
        throw new Error(`Manual control request failed with status ${response.status}`);
    }
    return response;
}

export async function fetchStatsReport() {
    const response = await fetch(`${BACKEND_URL}/stats_report`);
    if (response.status === 204) return null;
    return response.json();
}

export async function fetchSpotConfig() {
    try {
        const response = await fetch(`${BACKEND_URL}/spot_config`, { cache: 'no-store' });
        if (!response.ok) return null;
        return await response.json();
    } catch {
        return null;
    }
}

export async function updateSpotConfig(config) {
    return fetch(`${BACKEND_URL}/spot_config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
    });
}

export async function fetchGestureConfig() {
    try {
        const response = await fetch(`${BACKEND_URL}/gesture_config`, { cache: 'no-store' });
        if (!response.ok) return null;
        return await response.json();
    } catch {
        return null;
    }
}

export async function updateGestureConfig(config) {
    return fetch(`${BACKEND_URL}/gesture_config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
    });
}

export async function fetchHasControl() {
    try {
        const response = await fetch(`${BACKEND_URL}/has_control`, { cache: 'no-store' });
        if (!response.ok) return null;
        return await response.json();
    } catch {
        return null;
    }
}

export async function updateHasControl(has_control) {
    return fetch(`${BACKEND_URL}/has_control`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ has_control })
    });
}

export async function fetchStaticGestureConfig() {
    try {
        const response = await fetch(`${BACKEND_URL}/static_gesture_config`, { cache: 'no-store' });
        if (!response.ok) return null;
        return await response.json();
    } catch {
        return null;
    }
}

export async function updateStaticGestureConfig(config) {
    return fetch(`${BACKEND_URL}/static_gesture_config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
    });
}

export async function fetchTwoHandConfig() {
    try {
        const response = await fetch(`${BACKEND_URL}/two_hand_config`, { cache: 'no-store' });
        if (!response.ok) return null;
        return await response.json();
    } catch {
        return null;
    }
}

export async function updateTwoHandConfig(config) {
    return fetch(`${BACKEND_URL}/two_hand_config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
    });
}
