#!/bin/bash

# Check if notebook name is provided
if [ -z "$1" ]; then
    echo "Error: Please provide a notebook name"
    echo "Usage: $0 <notebook_name.ipynb>"
    exit 1
fi

NOTEBOOK=$1
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGS_DIR="logs"
BASENAME=$(basename "$NOTEBOOK" .ipynb)
LOG_PATH="$LOGS_DIR/${BASENAME}_$TIMESTAMP.ipynb"

# Create logs directory if it doesn't exist
mkdir -p "$LOGS_DIR"

echo "Logging to $LOG_PATH..."
papermill --log-output $NOTEBOOK $LOG_PATH
