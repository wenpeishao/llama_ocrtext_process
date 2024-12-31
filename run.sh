#!/bin/bash

# Exit immediately if any command returns a non-zero exit code
set -e

################################################################################
# 1) Debug & Environment Checks
################################################################################

echo "Current working directory: $(pwd)"
echo "Contents of the current directory before copying:"
ls -lah

# Check writability
if [ ! -w "$(pwd)" ]; then
    echo "Warning: Current directory is not writable. Please check permissions."
else
    echo "Destination directory is writable."
fi

# Check /staging/wshao33
echo "Checking access to /staging/wshao33..."
if [ ! -d /staging/wshao33 ]; then
    echo "Warning: /staging/wshao33 directory does not exist or is not accessible."
else
    if [ ! -r /staging/wshao33 ]; then
        echo "Warning: /staging/wshao33 directory is not readable. Please check permissions."
    else
        echo "/staging/wshao33 is accessible."
        echo "Contents of /staging/wshao33:"
        ls -lah /staging/wshao33
    fi
fi

################################################################################
# 2) Copy Required Tar Files
################################################################################

# Copy best_llama_model.tar.gz from /staging/wshao33
echo "Copying best_llama_model.tar.gz from /staging/wshao33..."
if ! cp -v /staging/wshao33/best_llama_model.tar.gz ./; then
    echo "Error: Failed to copy best_llama_model.tar.gz."
    exit 1
fi

# Copy processed_data.tar.gz (adjust if it's elsewhere, e.g., /home/wshao33/staging)
echo "Copying processed_data.tar.gz from /home/wshao33/staging..."
if ! cp -v /staging/wshao33/processed_data.tar.gz ./; then
    echo "Error: Failed to copy processed_data.tar.gz."
    exit 1
fi

echo "All tar files copied successfully!"
echo "Contents of the current directory after copying:"
ls -lah

################################################################################
# 3) Extract the Tarballs
################################################################################

# Extract the model
if [ -f best_llama_model.tar.gz ]; then
    echo "Extracting best_llama_model.tar.gz..."
    tar -xzf best_llama_model.tar.gz
    echo "Extraction of best_llama_model completed."
else
    echo "Warning: best_llama_model.tar.gz is missing."
fi
rm -rf best_llama_model.tar.gz
# Extract the processed data
if [ -f processed_data.tar.gz ]; then
    echo "Extracting processed_data.tar.gz..."
    tar -xzf processed_data.tar.gz
    echo "Extraction of processed_data completed."
else
    echo "Warning: processed_data.tar.gz is missing."
fi

################################################################################
# 4) Run the Python Script (Processing / Inference)
################################################################################

# Adjust script name as needed
if [ -f "llama_train.py" ]; then
    echo "Running the Python script: llama_train.py..."
    if ! python llama_train.py; then
        echo "Error: Python script execution failed."
        exit 1
    fi
    echo "Python script completed."
else
    echo "No llama_train.py script found; skipping processing."
fi

################################################################################
# 5) Archive processed_data_with_predictions & Copy Back
################################################################################

# If your Python script places the final CSVs in ./processed_data_with_predictions,
# we now tar up that directory
if [ -d "processed_data_with_predictions" ]; then
    echo "Creating processed_data_with_predictions.tar.gz..."
    tar -czf processed_data_with_predictions.tar.gz processed_data_with_predictions
    echo "Tar archive created: processed_data_with_predictions.tar.gz"

    echo "Copying processed_data_with_predictions.tar.gz back to /staging/wshao33..."
    cp -v processed_data_with_predictions.tar.gz /staging/wshao33/
else
    echo "Warning: Directory processed_data_with_predictions not found."
fi

################################################################################
# 6) Cleanup: Remove Everything (including best_llama_model.tar.gz)
################################################################################

echo "Removing all local files and folders..."

# Remove tarballs
if [ -f best_llama_model.tar.gz ]; then
    rm -f best_llama_model.tar.gz
    echo "Removed best_llama_model.tar.gz."
fi
if [ -f processed_data.tar.gz ]; then
    rm -f processed_data.tar.gz
    echo "Removed processed_data.tar.gz."
fi
if [ -f processed_data_with_predictions.tar.gz ]; then
    rm -f processed_data_with_predictions.tar.gz
    echo "Removed processed_data_with_predictions.tar.gz."
fi

# Remove extracted directories
if [ -d best_llama_model ]; then
    rm -rf best_llama_model
    echo "Removed best_llama_model directory."
fi
if [ -d processed_data ]; then
    rm -rf processed_data
    echo "Removed processed_data directory."
fi
if [ -d processed_data_with_predictions ]; then
    rm -rf processed_data_with_predictions
    echo "Removed processed_data_with_predictions directory."
fi

echo "Contents of the current directory after cleanup:"
ls -lah

################################################################################
# 7) Done
################################################################################

echo "Job completed. Check /staging/wshao33 for processed_data_with_predictions.tar.gz."
