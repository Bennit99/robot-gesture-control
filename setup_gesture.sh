#!/bin/bash

PULL_CHANGES=false
for arg in "$@"; do
    if [[ "$arg" == "--update" || "$arg" == "-u" || "$arg" == "--pull" ]]; then
        PULL_CHANGES=true
    fi
done

if [ "$PULL_CHANGES" = true ]; then
    # 1. Connect to Wi-Fi (if not already connected)
    echo "=== Checking Internet Connection ==="
    if ping -c 1 8.8.8.8 &> /dev/null; then
        echo "Internet connection is already active."
    else
        echo "No internet connection detected. Connecting to eduroam..."
        sudo nmcli connection up eduroam --ask
    fi
fi

# 2. Mount the external drive (if not already mounted)
echo -e "\n=== Checking External Drive ==="
if mountpoint -q /mnt/extdrive; then
    echo "/mnt/extdrive is already mounted."
else
    echo "Mounting external drive..."
    if sudo mount -t exfat -o uid=1000,gid=1000 /dev/sda2 /mnt/extdrive 2>/dev/null; then
        echo "Successfully mounted /dev/sda2 to /mnt/extdrive"
    elif sudo mount -t exfat -o uid=1000,gid=1000 /dev/sda1 /mnt/extdrive 2>/dev/null; then
        echo "Successfully mounted /dev/sda1 to /mnt/extdrive"
    else
        echo "Error: Failed to mount both /dev/sda2 and /dev/sda1. Please check the drive."
        return 1 2>/dev/null || exit 1
    fi
fi

# 3. Navigate to the project directory
echo -e "\n=== Navigating to project directory ==="
cd /mnt/extdrive/bennit/robot-gesture-control || { 
    echo "Error: Could not find /mnt/extdrive/bennit/robot-gesture-control"
    return 1 2>/dev/null || exit 1 
}

GIT_OUTPUT="Already up to date."
if [ "$PULL_CHANGES" = true ]; then
    # 3.5 Configure Git Sparse-Checkout for Inference
    echo -e "\n=== Configuring Git Sparse-Checkout ==="
    git sparse-checkout init --cone
    # In cone mode, root files (main.py, start.sh, pyproject.toml, etc.) are always included.
    # Explicitly include the required directories:
    git sparse-checkout set cfg engine jester utils actions models data_loader plot docker

    # 4. Pull from Git and check if there were changes
    echo -e "\n=== Pulling from Git ==="
    GIT_OUTPUT=$(git pull 2>&1)
    PULL_STATUS=$?

    if [ $PULL_STATUS -ne 0 ]; then
        if echo "$GIT_OUTPUT" | grep -qi "unstaged changes"; then
            echo "Unstaged changes detected. Stashing changes..."
            git stash
            GIT_OUTPUT=$(git pull 2>&1)
            PULL_STATUS=$?
            git stash pop
        fi
    fi

    echo "$GIT_OUTPUT"

    if [ $PULL_STATUS -ne 0 ]; then
        echo "Error: Git pull failed. Aborting application start."
        return 1 2>/dev/null || exit 1
    fi
fi

# 5. Stop existing container if running
echo -e "\n=== Checking for existing container ==="
if docker ps -a --format '{{.Names}}' | grep -iq "gesture"; then
    echo "Stopping and removing existing gesture container(s)..."
    docker ps -a --format '{{.Names}}' | grep -i "gesture" | xargs -r docker stop 2>/dev/null || true
    docker ps -a --format '{{.Names}}' | grep -i "gesture" | xargs -r docker rm 2>/dev/null || true
fi

# 6. Start the application
echo -e "\n=== Starting Application ==="
if [[ "$GIT_OUTPUT" == *"Already up to date."* ]]; then
    echo "No new changes detected (or pull skipped). Starting normally..."
    bash start.sh --detach
else
    echo "New changes pulled! Rebuilding the Docker container..."
    bash start.sh --build --detach
fi

echo -e "\n\033[92m============================================================\033[0m"
echo -e "\033[92;1m  [Frontend] APP IS RUNNING IN BACKGROUND\033[0m"
echo -e "\033[92m  Access at (if running on Spot payload): http://192.168.10.3:31002/gesture\033[0m"
echo -e "\033[92m============================================================\033[0m\n"
