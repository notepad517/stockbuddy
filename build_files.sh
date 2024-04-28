#!/bin/bash

# Activate virtual environment
source ./virtualenv/bin/activate

# Install dependencies
python3 -m pip install -r requirements.txt

pip install psycopg2-binary==2.9.9


# Collect static files
python manage.py collectstatic --noinput

# Deactivate virtual environment
deactivate
