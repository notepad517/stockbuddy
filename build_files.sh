#!/bin/bash

# Activate virtual environment
source ./virtualenv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Collect static files
python3.9 manage.py collectstatic --noinput

# Deactivate virtual environment
deactivate
