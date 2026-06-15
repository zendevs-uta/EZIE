"""
ezie_utils.py

These are helper functions for the EZIE Mag project

This module currently handles:
- Unzipping a day's data archive
- Finding .smr.60s.txt summary files inside it
- Loading and combining them into a single, time-indexed pandas DataFrame

Designed to be imported into both the daily and, eventually, the weekly
Jupyter notebooks so the loading logic only lives in one place

Expected folder layout (in relation to the notebook):
    rawdata/    <- zipped daily archives (should remain untouched)
    extracted/  <- unzipped .smr.60s.txt files (intermediate, disposable)
    data/       <- more permanant, cleaned datasets for ML, not developed yet
"""

# Basic imports
import os # handles file paths and directory walking
import re # finds the date (YYYYMMDD) in zip filenames
import zipfile # reads/extracts zip archives
import pandas as pd # for reading the data into DataFrames

# ---------------------------------------------------------------------
# Column names for the EZIE-mag .smr.60s summary file format
#
# The raw .txt files have NO header row, so pandas has no way to know what each column means on its own.
# This list comes directly from the EZIE_Mag_Data_Format.docx file and must stay in this exact order, since
# pandas will just assign these names left-to-right to whatever columns it finds.
#
# Defined once here (rather than retyped in every function) so there's a single source -> if the format ever changes,
# you only update it in one place
# ---------------------------------------------------------------------
SMR_COLUMNS = [
    'timeString', 'tval', 'intt', 'nsamp', 'stid', 'fingerprint',
    'lat', 'lon', 'alt', 'tres', 'ctemp', 'ccr',
    'Bx', 'By', 'Bz', 'afs_sel', 'fs_sel',
    'Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'imu_ctemp'
]

def unzip_day(zip_path, extract_to):
    """
    Extract a daily EZIE Mag zip archive and return a list of paths to the .smr.60s.txt
    file(s) for that day.


    Parameters
    ---------------------------------------------------------------------
    zip_path : str
        Path to the .zip file (e.g. "rawdata/eziemag.20260605.AAAAAIzs1uUA.zip")
    extract_to : str
        Folder to extract the zip's contents into (e.g. "extracted").
        Will be created if it doesn't already exist. NOTE: the zip's
        internal folder structure (e.g. home/ezie/smr.60s/20260605/...)
        is preserved, so files will end up nested inside this folder,
        not directly in it. Files from previously-processed days will
        also remain here, but won't interfere since we go straight to
        this day's specific date folder.
    ---------------------------------------------------------------------


    Returns
    ---------------------------------------------------------------------
    list[str]
        Full paths to each .smr.60s.txt file for this day, sorted
        alphabetically. Because these filenames encode the
        date/time, alphabetical order is also chronological order.
    ---------------------------------------------------------------------
    """

    # *** Figure out which day's folder we're looking for ***
    # Every EZIE Mag zip, once extracted, follows the same internal
    # structure: home/ezie/smr.60s/<date>/eziemag.<...>.smr.60s.txt
    # where <date> is an 8-digit YYYYMMDD folder. We pull that date
    # out of the zip's filename so we know exactly where to look,
    # instead of searching the whole extracted/ folder (which would
    # also pick up files from other days extracted previously).

    # os.path.basename() strips off any folder path, leaving just the
    # filename itself, e.g. "eziemag.20260605.AAAAAIzs1uUA.zip"
    zip_name = os.path.basename(zip_path)

    # re.search() looks for a pattern anywhere in the string.
    # r"\d{8}" means "8 consecutive digits" - this matches a YYYYMMDD
    # date regardless of whether it's surrounded by dots, underscores,
    # or anything else.
    date_match = re.search(r"\d{8}", zip_name)

    if date_match is None:
        # If we can't find an 8-digit date in the filename, something
        # is wrong (unexpected naming convention) - better to fail
        # loudly here than to silently look in the wrong folder.
        raise ValueError(
            f"Could not find an 8-digit date (YYYYMMDD) in zip filename: {zip_name}"
        )

    date_str = date_match.group(0)  # e.g. "20260605"

    # Make sure the extraction folder exists.
    # exist_ok=True means: if the folder is already there, don't raise an error, just continue.
    # This makes it safe to call this function repeatedly (e.g. once per day) without it
    # complaining the folder already exists.
    os.makedirs(extract_to, exist_ok=True)

    # Open the zip file in read mode ('r').
    # The "with" statement automatically closes the zip file when we're done, even if something goes wrong inside the block.
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # extractall() pulls every file out of the zip and writes it to disk under extract_to, recreating whatever folder
        # structure was stored inside the zip itself.
        zf.extractall(extract_to)

    # *** Go directly to this day's smr.60s folder ***
    # Build the path to where this day's summary files live, e.g.:
    #   extracted/home/ezie/smr.60s/20260605
    day_folder = os.path.join(extract_to, "home", "ezie", "smr.60s", date_str)

    if not os.path.isdir(day_folder):
        # If the expected folder doesn't exist, the zip's internal
        # structure didn't match what we assumed - fail loudly rather
        # than silently returning an empty list.
        raise FileNotFoundError(
            f"Expected folder not found after extraction: {day_folder}"
        )

    # List every file in that folder, keeping only the summary files.
    # This automatically skips the .rawz raw data files (and anything
    # else), since their names don't end in ".smr.60s.txt". .rawz data
    # files are only needed for making the smr.60 files afaik.
    smr_files = []
    for f in os.listdir(day_folder):
        if f.endswith('.smr.60s.txt'):
            smr_files.append(os.path.join(day_folder, f))

    # Sort alphabetically. Since these filenames contain YYYYMMDDHH timestamps, sorting alphabetically also sorts
    # them in chronological order.
    return sorted(smr_files)


def load_smr_files(file_paths):
    """
    Load one or more .smr.60s.txt files into a single DataFrame.

    Steps performed:
      - Read each file as whitespace-separated values with no header
      - Assign the 24 standard EZIE-Mag column names (defined at top)
      - Stack all files together into one DataFrame
      - Convert the 'timeString' column into real datetime objects
      - Set that datetime column as the DataFrame's index, and sort by it
        (so everything is in time order, even if files were passed in out of order)


    Parameters
    ---------------------------------------------------------------------
    file_paths : list[str] or str
        One or more paths to .smr.60s.txt files. A single string is
        also accepted for convenience (e.g. when there's only one
        file for the day).
    ---------------------------------------------------------------------
    

    Returns
    ---------------------------------------------------------------------
    pandas.DataFrame
        Combined DataFrame with all 24 EZIE-Mag columns, indexed by
        timestamp (timeString), sorted chronologically.
    ---------------------------------------------------------------------
    """

    # *** Convenience handling ***
    # If the caller passed a single filename (a string) instead of a list, wrap it in a list so the loop below works the same either
    # way (looping over a list of 1 item vs many items).
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    # This list will hold one DataFrame per file. We'll combine them all at the end.
    dfs = []

    for path in file_paths:
        # Read the file into a DataFrame:
        #   sep=r"\s+"     -> split on any amount of whitespace
        #                      (handles single spaces, multiple
        #                      spaces, or tabs between values)
        #   engine="python" -> required for regex separators like
        #                      r"\s+"; slower than the default C
        #                      engine but necessary here
        #   header=None    -> the file has NO header row, so don't
        #                      let pandas try to use the first line
        #                      as column names
        df = pd.read_csv(
            path,
            sep=r"\s+",
            engine="python",
            header=None
        ) # standard csv read, append to dfs later

        # Assign the known column names (in order) from SMR_COLUMNS. This must match the actual number/order of columns in the
        # file exactly, or columns will be mislabeled.
        df.columns = SMR_COLUMNS

        # Add this file's DataFrame to our list.
        dfs.append(df)

    # Stack all the individual file DataFrames into one big DataFrame. ignore_index=True means: don't keep each file's original row
    # numbers (0,1,2...) - instead renumber every row continuously across all files (0,1,2,...,N). Without this, you'd get
    # duplicate index values if multiple files each start at row 0.
    full_df = pd.concat(dfs, ignore_index=True)

    # *** Convert timestamps from text to real datetime objects ***
    # Right now 'timeString' is just a string like "2026-06-05T23:58:38.2149494Z". pd.to_datetime() parses that
    # into an actual datetime, which pandas can use for time-based plotting, sorting, gap detection, etc.
    full_df['timeString'] = pd.to_datetime(full_df['timeString'])

    # *** Use the timestamp as the row index ***
    # set_index('timeString') makes the timestamp the "label" for each row instead of a plain integer. This is what lets
    # matplotlib plot against real time on the x-axis.
    #
    # sort_index() then sorts all rows by that timestamp - useful if files were loaded out of order, or if multiple files overlap.
    full_df = full_df.set_index('timeString').sort_index()

    # Return the finished, ready-to-plot DataFrame.
    return full_df