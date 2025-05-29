#!/bin/bash

# Navigate to dashboard directory
cd "$(dirname "$0")"

# Check if virtual environment exists, if not create it
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
    source venv/bin/activate
    echo "Installing requirements..."
    pip install -r requirements.txt
else
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Run the Streamlit app
echo "Starting Financial Crisis Early Warning Dashboard..."
streamlit run app.py 