"""
ezie_utils.py

These are helper functions for the EZIE Mag project

This module currently handles:
- Unzipping a day's data archive
- Finding .smr.60s.txt summary files inside it
- Loading and combining them into a single, time-indexed pandas DataFrame
- Rounding timestamps and filling gaps in missing data
- Computing derived magnetic field features (B_total, Bh, declination, etc)
- Identifying which physical device a row of data came from
- Saving / loading processed daily CSVs for later weekly/monthly use

Designed to be imported into the Processing, Daily, Weekly, and Monthly Jupyter notebooks so this
logic only lives in one place. 

Expected folder layout (relative to the project root, NOT the notebooks/ folder - notebooks add the
project root to sys.path so this import works correctly regardless of where the notebook itself lives):

    data/
        raw/        <- zipped daily archives (should remain untouched)
        interim/    <- unzipped .smr.60s.txt files (scratch space, disposable)
        processed/  <- cleaned, feature-engineered daily CSVs (the
                        source of truth for all downstream analysis)
    ezie_utils.py
    notebooks/
        Data_Processing.ipynb
        Daily.ipynb
        ...

NOTE ON SCOPE: this module intentionally does NOT do outlier/QC flagging or imputation. Per current
project decisions, that logic lives directly in the Processing notebook (not here) until an
imputation strategy is designed. This module only handles mechanical, deterministic steps - loading,
gap-filling, feature engineering, and saving/loading. Affine calibration to the FRD/NED reference frame is
also not implemented here yet - data stays in raw EZIE sensor units.
"""

# --- Standard library imports ---
import os       # handles file paths and directory walking
import re       # finds the date (YYYYMMDD) in zip filenames
import glob     # finds files matching a wildcard pattern
import zipfile  # reads/extracts zip archives

# --- Third-party imports ---
import numpy as np     # for sqrt / arctan2 in derived field calculations
import pandas as pd    # for DataFrames and time-based operations

# ----------------------------------------------------------------------------------------------------
# Column names for the EZIE-mag .smr.60s summary file format
#
# The raw .txt files have NO header row, so pandas has no way to know what each column means on its own.
# This list comes directly from the EZIE_Mag_Data_Format.docx file and must stay in this exact order,
# since pandas will just assign these names left-to-right to whatever columns it finds.
#
# Defined once here (rather than retyped in every function) so there's
# a single source -> if the format ever changes, you only update it in
# one place.
# ----------------------------------------------------------------------------------------------------
SMR_COLUMNS = [
    'timeString', 'tval', 'intt', 'nsamp', 'stid', 'fingerprint',
    'lat', 'lon', 'alt', 'tres', 'ctemp', 'ccr',
    'Bx', 'By', 'Bz', 'afs_sel', 'fs_sel',
    'Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'imu_ctemp'
]


# ----------------------------------------------------------------------------------------------------
# Known device fingerprints, mapped to a human-readable name/location.
#
# 'fingerprint' is the 12-character ID that uniquely identifies the physical EZIE sensor a row of data
# came from. This matters once there's more than one device (e.g. collaborator sites) - it lets you
# tell devices apart, and matters for calibration since the affine transform (A, b) is specific to one
# physical sensor's mounting and electronics.
#
# Add new devices here as collaborators come online. 'stid' in the raw data appears to just be the
# generic string "eziemag" rather than a per-device identifier - this should be confirmed against real
# data (df['stid'].unique()) but fingerprint is treated as the reliable per-device key.
# ----------------------------------------------------------------------------------------------------
STATION_NAMES = {
    "AAAAAIzs1uUA": "Zane (Keller, TX)",
    # "<fingerprint>": "<collaborator name/location>",
}


def unzip_day(zip_path, extract_to):
    """
    Extract a daily EZIE Mag zip archive and return a list of paths to the .smr.60s.txt file(s)
    for that day.

    Parameters
    --------------------------------------------------------------------------------
    zip_path : str
        Path to the .zip file (e.g. "data/raw/eziemag.20260605.AAAAAIzs1uUA.zip")
    extract_to : str
        Folder to extract the zip's contents into (e.g. "data/interim"). Will be
        created if it doesn't already exist. NOTE: the zip's internal folder
        structure (e.g. home/ezie/smr.60s/20260605/...) is preserved, so files will
        end up nested inside this folder, not directly in it. Files from
        previously-processed days will also remain here, but won't interfere since
        we go straight to this day's specific date folder.
    --------------------------------------------------------------------------------

    Returns
    --------------------------------------------------------------------------------
    list[str]
        Full paths to each .smr.60s.txt file for this day, sorted alphabetically.
        Because these filenames encode the date/time, alphabetical order is also
        chronological order.
    --------------------------------------------------------------------------------
    """

    # *** Figure out which day's folder we're looking for ***
    # Every EZIE Mag zip, once extracted, follows the same internal
    # structure: home/ezie/smr.60s/<date>/eziemag.<...>.smr.60s.txt
    # where <date> is an 8-digit YYYYMMDD folder. We pull that date out
    # of the zip's filename so we know exactly where to look, instead
    # of searching the whole extract_to folder (which would also pick
    # up files from other days extracted previously).

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
    # exist_ok=True means: if the folder is already there, don't raise
    # an error, just continue. This makes it safe to call this function
    # repeatedly (e.g. once per day) without it complaining the folder
    # already exists.
    os.makedirs(extract_to, exist_ok=True)

    # Open the zip file in read mode ('r').
    # The "with" statement automatically closes the zip file when we're
    # done, even if something goes wrong inside the block.
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # extractall() pulls every file out of the zip and writes it to
        # disk under extract_to, recreating whatever folder structure
        # was stored inside the zip itself.
        zf.extractall(extract_to)

    # *** Go directly to this day's smr.60s folder ***
    # Build the path to where this day's summary files live, e.g.:
    #   data/interim/home/ezie/smr.60s/20260605
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

    # Sort alphabetically. Since these filenames contain YYYYMMDDHH
    # timestamps, sorting alphabetically also sorts them in
    # chronological order.
    return sorted(smr_files)


def load_smr_files(file_paths):
    """
    Load one or more .smr.60s.txt files into a single DataFrame.

    Steps performed:
      - Read each file as whitespace-separated values with no header
      - Assign the 24 standard EZIE-Mag column names (defined at top)
      - Stack all files together into one DataFrame
      - Convert the 'timeString' column into real datetime objects
      - Set that datetime column as the DataFrame's index, and sort by
        it (so everything is in time order, even if files were passed
        in out of order)

    Parameters
    ---------------------------------------------------------------------
    file_paths : list[str] or str
        One or more paths to .smr.60s.txt files. A single string is
        also accepted for convenience (e.g. when there's only one file
        for the day).
    ---------------------------------------------------------------------

    Returns
    ---------------------------------------------------------------------
    pandas.DataFrame
        Combined DataFrame with all 24 EZIE-Mag columns, indexed by
        timestamp (timeString), sorted chronologically.
    ---------------------------------------------------------------------
    """

    # *** Convenience handling ***
    # If the caller passed a single filename (a string) instead of a
    # list, wrap it in a list so the loop below works the same either
    # way (looping over a list of 1 item vs many items).
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    # This list will hold one DataFrame per file. We'll combine them
    # all at the end.
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
        )  # standard csv read, append to dfs later

        # Assign the known column names (in order) from SMR_COLUMNS.
        # This must match the actual number/order of columns in the
        # file exactly, or columns will be mislabeled.
        df.columns = SMR_COLUMNS

        # Add this file's DataFrame to our list.
        dfs.append(df)

    # Stack all the individual file DataFrames into one big DataFrame.
    # ignore_index=True means: don't keep each file's original row
    # numbers (0,1,2...) - instead renumber every row continuously
    # across all files (0,1,2,...,N). Without this, you'd get
    # duplicate index values if multiple files each start at row 0.
    full_df = pd.concat(dfs, ignore_index=True)

    # *** Convert timestamps from text to real datetime objects ***
    # Right now 'timeString' is just a string like
    # "2026-06-05T23:58:38.2149494Z". pd.to_datetime() parses that into
    # an actual datetime, which pandas can use for time-based plotting,
    # sorting, gap detection, etc.
    full_df['timeString'] = pd.to_datetime(full_df['timeString'])

    # *** Use the timestamp as the row index ***
    # set_index('timeString') makes the timestamp the "label" for each
    # row instead of a plain integer. This is what lets matplotlib plot
    # against real time on the x-axis.
    #
    # sort_index() then sorts all rows by that timestamp - useful if
    # files were loaded out of order, or if multiple files overlap.
    full_df = full_df.set_index('timeString').sort_index()

    # Return the finished DataFrame, ready for gap-filling.
    return full_df


def round_and_fill_gaps(df):
    """
    Round timestamps to the nearest minute and reindex the DataFrame
    onto a complete 1-minute grid spanning the full calendar day.

    Background
    -----------
    Raw timestamps have sub-second precision that drifts slightly
    minute to minute (e.g. ":22.020162600", ":22.038212300", ...) due
    to small clock/sampling jitter in the device. This means the real
    timestamps don't land on perfectly even 60-second boundaries.

    If data collection stops partway through the day (e.g. a power
    outage), the file simply has fewer rows - there's no placeholder
    for the missing minutes. Plotting this as-is would draw a
    misleading straight line connecting the last reading before a gap
    to the first reading after it.

    This function fixes both issues:
      1. Rounds each timestamp to its nearest whole minute, so real
         readings align to a clean grid
      2. Builds a complete 1-minute-interval index covering the full
         calendar day (00:00 to 23:59) and reindexes onto it - any
         minute with no real reading becomes a row of NaN, which
         matplotlib will correctly render as a break in line plots

    Parameters
    ---------------------------------------------------------------------
    df : pandas.DataFrame
        A DataFrame as returned by load_smr_files(), with a datetime
        index (not yet rounded or gap-filled).
    ---------------------------------------------------------------------

    Returns
    ---------------------------------------------------------------------
    pandas.DataFrame
        A new DataFrame reindexed onto a complete 1-minute grid for the
        calendar day of df's first timestamp. Rows with no original
        data are filled with NaN across all columns.
    ---------------------------------------------------------------------
    """

    # Work on a copy so we don't modify the caller's original DataFrame
    # in place - safer when this function might be called more than
    # once or the original df is still needed elsewhere.
    df = df.copy()

    # --- Step 1: round each timestamp to the nearest minute ---
    # This absorbs the small sub-second jitter so real readings line
    # up exactly with the clean 1-minute grid built in Step 2.
    df.index = df.index.round("1min")

    # --- Step 1b: guard against duplicate timestamps after rounding ---
    # If two readings happen to round to the same minute, reindex()
    # below would fail on duplicate index labels. This keeps the
    # first occurrence and drops any later duplicates.
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="first")]

    # --- Step 2: build a complete 1-minute grid for the whole day ---
    # df.index.min().floor("D") rounds the first timestamp DOWN to
    # midnight (00:00:00) of that day - this is the grid's start point.
    # Adding 23 hours 59 minutes gives the grid's end point (23:59).
    # freq="1min" means one timestamp per minute in between.
    # This produces exactly 1440 timestamps (24 hours x 60 minutes).
    day_start = df.index.min().floor("D")
    full_day_index = pd.date_range(
        start=day_start,
        end=day_start + pd.Timedelta(hours=23, minutes=59),
        freq="1min"
    )

    # --- Step 3: reindex onto that grid ---
    # Rows that exist in df keep their values; minutes with no
    # matching timestamp become new rows filled with NaN.
    df = df.reindex(full_day_index)
    df.index.name = "timeString"

    return df


def add_derived_fields(df):
    """
    Add derived magnetic field features to a DataFrame that already has
    Bx, By, Bz columns.

    Adds:
      - B_total : total field magnitude, sqrt(Bx^2 + By^2 + Bz^2)
                  Rotation-independent overall field strength; the
                  quantity most directly comparable to an observatory's
                  reported F value.
      - Bh      : horizontal field magnitude, sqrt(Bx^2 + By^2)
                  Combines the North/East components only (excludes
                  Down); standard geomagnetic "H" quantity.
      - D       : declination, in degrees, atan2(By, Bx)
                  The angle between magnetic North and the sensor's
                  measured horizontal field direction. Standard
                  geomagnetic "D" quantity, directly comparable to an
                  observatory's reported D value.

    NaN handling: if Bx, By, or Bz is NaN for a given row (e.g. a
    gap-filled row from round_and_fill_gaps()), any arithmetic
    involving NaN also produces NaN. So all three derived fields will
    correctly be NaN during data gaps, consistent with the raw
    columns - matplotlib will break the line there too.

    Parameters
    ---------------------------------------------------------------------
    df : pandas.DataFrame
        Must contain 'Bx', 'By', and 'Bz' columns (nT).
    ---------------------------------------------------------------------

    Returns
    ---------------------------------------------------------------------
    pandas.DataFrame
        A new DataFrame with 'B_total', 'Bh', and 'D' columns added.
    ---------------------------------------------------------------------
    """

    df = df.copy()

    # Total field magnitude - combines all three axes, independent of
    # sensor orientation.
    df["B_total"] = np.sqrt(df["Bx"]**2 + df["By"]**2 + df["Bz"]**2)

    # Horizontal field magnitude - North/East components only.
    df["Bh"] = np.sqrt(df["Bx"]**2 + df["By"]**2)

    # Declination - angle of the horizontal field from the sensor's
    # +X axis, in degrees. np.arctan2 handles all four quadrants
    # correctly (unlike a plain arctan(By/Bx), which would not).
    df["D"] = np.degrees(np.arctan2(df["By"], df["Bx"]))

    return df


def find_zip_for_date(date_str, raw_dir):
    """
    Find the zip file in raw_dir whose filename contains the given
    date.

    Validates that date_str is a real 8-digit YYYYMMDD date, then
    searches raw_dir for a matching zip file. Raises a clear error if
    the date is malformed, no file is found, or more than one file
    matches.

    Parameters
    ---------------------------------------------------------------------
    date_str : str
        Date in YYYYMMDD format, e.g. "20260609".
    raw_dir : str
        Folder containing the raw zip files (e.g. "data/raw").
    ---------------------------------------------------------------------

    Returns
    ---------------------------------------------------------------------
    str
        Full path to the matching zip file.
    ---------------------------------------------------------------------
    """

    # --- Validate the date is exactly 8 digits ---
    if not (date_str.isdigit() and len(date_str) == 8):
        raise ValueError(
            f"Invalid date '{date_str}'. Please provide exactly 8 digits in YYYYMMDD format (e.g. 20260609)."
        )

    # --- Validate it's an actual calendar date ---
    # strptime() raises ValueError on its own for things like month=13
    # or day=32; we let that propagate with a clearer message.
    import datetime
    try:
        datetime.datetime.strptime(date_str, "%Y%m%d")
    except ValueError:
        raise ValueError(
            f"'{date_str}' is not a valid calendar date. Please provide a real date in YYYYMMDD format."
        )

    # --- Search for a matching zip file ---
    # glob with wildcards on either side of date_str matches the date
    # regardless of separator style (dots, underscores, etc.)
    matches = glob.glob(os.path.join(raw_dir, f"*{date_str}*.zip"))

    if len(matches) == 0:
        raise FileNotFoundError(f"No zip file found for date {date_str} in {raw_dir}/")
    elif len(matches) > 1:
        raise ValueError(f"Multiple zip files match date {date_str}: {matches}")

    return matches[0]


def save_processed_day(df, date_str, processed_dir, overwrite=False):
    """
    Save a processed (gap-filled, feature-engineered) daily DataFrame
    to processed_dir as a CSV.

    Does NOT silently overwrite an existing file - if a file for this
    date already exists and overwrite=False, it returns a status of
    "exists" without writing anything, so the calling notebook can
    decide whether to prompt the user before overwriting.

    Parameters
    ---------------------------------------------------------------------
    df : pandas.DataFrame
        The processed DataFrame to save, indexed by timestamp.
    date_str : str
        Date in YYYYMMDD format, used to build the output filename.
    processed_dir : str
        Folder to save the CSV into (e.g. "data/processed"). Will be
        created if it doesn't already exist.
    overwrite : bool, default False
        If True, overwrite an existing file for this date without
        being asked. If False (default), an existing file is left
        untouched and a status of "exists" is returned instead.
    ---------------------------------------------------------------------

    Returns
    ---------------------------------------------------------------------
    dict
        {
            "status": "saved" | "exists",
            "path": str  # full path to the (would-be) output file
        }
    ---------------------------------------------------------------------
    """

    os.makedirs(processed_dir, exist_ok=True)

    output_path = os.path.join(processed_dir, f"eziemag_{date_str}_processed.csv")

    # If the file already exists and we're not allowed to overwrite,
    # stop here and tell the caller - this lets the Processing
    # notebook ask the user before clobbering a previous version.
    if os.path.exists(output_path) and not overwrite:
        return {"status": "exists", "path": output_path}

    df.to_csv(output_path)

    return {"status": "saved", "path": output_path}


def load_processed_days(date_list, processed_dir):
    """
    Load and concatenate previously-saved processed daily CSVs (as
    written by save_processed_day()).

    Missing files (a date with no processed CSV yet) are skipped with
    a printed warning rather than raising an error, so a week or month
    with a few gaps still loads successfully.

    Parameters
    ---------------------------------------------------------------------
    date_list : list[str]
        List of YYYYMMDD date strings to load.
    processed_dir : str
        Folder containing the processed CSVs (e.g. "data/processed").
    ---------------------------------------------------------------------

    Returns
    ---------------------------------------------------------------------
    pandas.DataFrame
        Combined DataFrame, sorted by time, spanning all successfully
        loaded days. Returns an empty DataFrame if no files were found.
    ---------------------------------------------------------------------
    """

    dfs = []

    for date_str in date_list:
        path = os.path.join(processed_dir, f"eziemag_{date_str}_processed.csv")

        if not os.path.exists(path):
            print(f"Warning: no processed file found for {date_str}, skipping ({path})")
            continue

        day_df = pd.read_csv(path, index_col="timeString", parse_dates=True)
        dfs.append(day_df)

    if len(dfs) == 0:
        print("Warning: no processed files were found for any requested date.")
        return pd.DataFrame()

    combined = pd.concat(dfs)
    return combined.sort_index()