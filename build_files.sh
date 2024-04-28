#!/bin/bash

# Activate virtual environment
source ./virtualenv/bin/activate

# Install dependencies
python3 -m pip install -r requirements.txt

# Collect static files
python manage.py collectstatic --noinput

# Deactivate virtual environment
deactivate
