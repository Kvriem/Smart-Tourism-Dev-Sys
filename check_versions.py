import pandas as pd
import sqlalchemy
import sys

print("Python version:", sys.version)
print("Pandas version:", pd.__version__)
print("SQLAlchemy version:", sqlalchemy.__version__)

# Check if we're using SQLAlchemy 2.0+
major_version = int(sqlalchemy.__version__.split('.')[0])
if major_version >= 2:
    print("WARNING: Using SQLAlchemy 2.0+, which has breaking changes")
else:
    print("Using SQLAlchemy 1.x")
