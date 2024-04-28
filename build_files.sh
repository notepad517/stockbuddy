#!/bin/bash

# Activate virtual environment
source ./venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Collect static files
python manage.py collectstatic --noinput

# Deactivate virtual environment
deactivate
