#!/bin/bash

# Ensure the script stops on first error
set -e

# Define paths
SERVER_DATA_PATH="/lustre/groups/ml01/workspace/kemal.inecik/tardis_data"
LOCAL_DATA_PATH="/Users/kemalinecik/git_nosync/tardis/data"
DRIVE_DATA_PATH="gdrive:/tardis/tardis_data"
REPO_PATH="/Users/kemalinecik/git_nosync/tardis"
DRIVE_REPO_PATH="gdrive:/tardis/repo"

# Define common exclude parameters as an array
COMMON_EXCLUDES=(--exclude '__pycache__/**' \
                 --exclude '*.DS_Store' \
                 --exclude '.idea/**' \
                 --exclude '.mypy_cache/**' \
                 --exclude '*.ipynb_checkpoints/**' \
                 --exclude '.git/**')

# Loop indefinitely until the user chooses to exit
while true; do
    # Display the menu to the user
    echo "Select an operation to perform:"
    echo "1. Send Data from Server to Drive"
    echo "2. Send Data from Local to Drive"
    echo "3. Send Data from Drive to Local"
    echo "4. Send Data from Drive to Server"
    echo "5. Sync Repo from Local to Drive"
    echo "q. Exit the script"

    # Read the user's choice
    read -p "Enter your choice (1-5): " choice

    # Execute the corresponding command based on the user's choice
    case $choice in
        1) rclone sync -v --progress -L "$SERVER_DATA_PATH" "$DRIVE_DATA_PATH" "${COMMON_EXCLUDES[@]}"  --exclude '*_temporary/**'
           ;;
        2) rclone sync -v --progress "$LOCAL_DATA_PATH" "$DRIVE_DATA_PATH" "${COMMON_EXCLUDES[@]}" --exclude '*_temporary/**'
           ;;
        3) rclone sync -v --progress "$DRIVE_DATA_PATH" "$LOCAL_DATA_PATH" "${COMMON_EXCLUDES[@]}" --exclude '*_temporary/**'
           ;;
        4) rclone sync -v --progress "$DRIVE_DATA_PATH" "$SERVER_DATA_PATH" "${COMMON_EXCLUDES[@]}"  --exclude '*_temporary/**'
           ;;
        5) rclone sync -v --progress "$REPO_PATH" "$DRIVE_REPO_PATH" "${COMMON_EXCLUDES[@]}" \
            --exclude '/data/**'
           ;;
        q) echo "Exiting script."
           exit 0
           ;;
        *) echo "Invalid choice. Please enter a number between 1 and 5."
           ;;
    esac
    echo "" # Print a newline for better readability before the menu is shown again
done
