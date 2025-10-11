# Streamlit Player Analytics Dashboard (Filtered Version)
# This version excludes users with 'test' in their email
# and only includes users from 'United States'.

import re
from datetime import datetime, date
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ... [All helper functions for loading/cleaning remain unchanged] ...

# After all filters and before KPI calculations:

# Apply additional exclusion and inclusion filters
fdf = fdf.copy()

# Exclude users with 'test' in email (case-insensitive)
if 'email' in fdf.columns:
    fdf = fdf[~fdf['email'].str.contains('test', case=False, na=False)]

# Include only users with country == 'United States' (case-insensitive match)
if 'country' in fdf.columns:
    fdf = fdf[fdf['country'].astype(str).str.lower().str.strip() == 'united states']

# Continue with KPI calculations and charts as before
# (Everything else in your dashboard remains the same.)
