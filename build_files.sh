#!/bin/bash

# Create a temporary virtual environment
python3 -m venv tempenv

# Activate the temporary virtual environment
source tempenv/bin/activate

# Install dependencies
python3 -m pip install -r requirements.txt

# Collect static files
python3 manage.py collectstatic --noinput

# Deactivate the temporary virtual environment
deactivate

# Remove the temporary virtual environment
rm -rf tempenv
