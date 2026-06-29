"""
ezie_utils.py

These are helper functions for the EZIE Mag project

This module currently handles:
- Unzipping daily or weekly data archives
- Finding .smr.60s.txt summary files inside them
- Loading and combining them into a single, time-indexed pandas DataFrame
- Rounding timestamps and filling gaps in missing data
- Computing derived magnetic field features (B_total, Bh, declination)
- Identifying which physical device a row of data came from
- Saving / loading processed daily CSVs organized by device fingerprint

Designed to be imported into the Processing, Daily, Weekly, and Monthly
Jupyter notebooks so this logic only lives in one place.

Expected folder layout (relative to the project root):

    data/
        raw/        <- zipped daily or weekly archives (should remain untouched)
        interim/    <- unzipped .smr.60s.txt files (scratch space, disposable)
        processed/
            <fingerprint>/   <- one subfolder per device
                eziemag_YYYYMMDD_<fingerprint>_processed.csv
    ezie_utils.py
    notebooks/
        Data_Processing.ipynb
        Daily.ipynb
        Multi-Day.ipynb
        ...

NOTE ON SCOPE: this module intentionally does NOT do outlier/QC
flagging or imputation. Per current project decisions, that logic
lives directly in the Processing notebook (not here) until an
imputation strategy is designed. This module only handles mechanical,
deterministic steps - loading, gap-filling, feature engineering, and
saving/loading. Affine calibration to the FRD/NED reference frame is
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


# ---------------------------------------------------------------------
# Column names for the EZIE-mag .smr.60s summary file format
#
# The raw .txt files have NO header row, so pandas has no way to know
# what each column means on its own. This list comes directly from the
# EZIE_Mag_Data_Format.docx file and must stay in this exact order,
# since pandas will just assign these names left-to-right to whatever
# columns it finds.
#
# Defined once here (rather than retyped in every function) so there's
# a single source -> if the format ever changes, you only update it in
# one place.
# ---------------------------------------------------------------------
SMR_COLUMNS = [
    'timeString', 'tval', 'intt', 'nsamp', 'stid', 'fingerprint',
    'lat', 'lon', 'alt', 'tres', 'ctemp', 'ccr',
    'Bx', 'By', 'Bz', 'afs_sel', 'fs_sel',
    'Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'imu_ctemp'
]


# ---------------------------------------------------------------------
# Known device fingerprints, mapped to a human-readable name/location.
#
# 'fingerprint' is the 12-character ID that uniquely identifies the
# physical EZIE sensor a row of data came from. This matters once
# there's more than one device (e.g. collaborator sites) - it lets you
# tell devices apart, and matters for calibration since the affine
# transform (A, b) is specific to one physical sensor's mounting and
# electronics.
#
# Add new devices here as collaborators come online.
# ---------------------------------------------------------------------
STATION_NAMES = {
    "AAAAAIzs1uUA": "Zane (Keller, TX)",
    # "<fingerprint>": "<collaborator name/location>",
}


def unzip_day(zip_path, extract_to):
    """
    Extract a daily EZIE Mag zip archive and return a list of paths to
    the .smr.60s.txt file(s) for that day.

    Parameters
    ---------------------------------------------------------------------
    zip_path : str
        Path to the .zip file (e.g. "data/raw/eziemag.20260605.AAAAAIzs1uUA.zip")
    extract_to : str
        Folder to extract the zip's contents into (e.g. "data/interim").
        Will be created if it doesn't already exist. The zip's internal
        folder structure (e.g. home/ezie/smr.60s/20260605/...) is
        preserved. Files from previously-processed days remain here but
        won't interfere since we go straight to this day's date folder.
    ---------------------------------------------------------------------

    Returns
    ---------------------------------------------------------------------
    list[str]
        Full paths to each .smr.60s.txt file for this day, sorted
        alphabetically (which also sorts them chronologically).
    ---------------------------------------------------------------------
    """

    # Pull the date from the zip filename (8 consecutive digits = YYYYMMDD)
    zip_name = os.path.basename(zip_path)
    date_match = re.search(r"\d{8}", zip_name)

    if date_match is None:
        raise ValueError(
            f"Could not find an 8-digit date (YYYYMMDD) in zip filename: {zip_name}"
        )

    date_str = date_match.group(0)

    # Extract the zip
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_to)

    # Go directly to this day's smr.60s folder
    day_folder = os.path.join(extract_to, "home", "ezie", "smr.60s", date_str)

    if not os.path.isdir(day_folder):
        raise FileNotFoundError(
            f"Expected folder not found after extraction: {day_folder}"
        )

    # List only .smr.60s.txt files, skipping .rawz and anything else
    smr_files = []
    for f in os.listdir(day_folder):
        if f.endswith('.smr.60s.txt'):
            smr_files.append(os.path.join(day_folder, f))

    return sorted(smr_files)


def unzip_week(zip_path, extract_to):
    """
    Extract a weekly EZIE Mag zip archive and return a dict mapping
    each date found inside to its list of .smr.60s.txt files.

    Extracts into a zip-specific subfolder under extract_to (named
    after the zip file, without the .zip extension) so that previously
    extracted daily zips in the same interim folder don't contaminate
    the results.

    Parameters
    ---------------------------------------------------------------------
    zip_path : str
        Path to the weekly .zip file
        (e.g. "data/raw/eziemag_20260622-20260628_AAAAAIzs1uUA.zip")
    extract_to : str
        Root interim folder (e.g. "data/interim"). The zip will be
        extracted into a subfolder named after the zip file itself,
        e.g. "data/interim/eziemag_20260622-20260628_AAAAAIzs1uUA/"
    ---------------------------------------------------------------------

    Returns
    ---------------------------------------------------------------------
    dict[str, list[str]]
        A dict mapping each date string (YYYYMMDD) found in the zip to
        a sorted list of full paths to its .smr.60s.txt files.
    ---------------------------------------------------------------------
    """

    # Extract into a zip-specific subfolder so previously extracted
    # daily zips in interim/ don't contaminate the results
    zip_name = os.path.splitext(os.path.basename(zip_path))[0]
    zip_extract_dir = os.path.join(extract_to, zip_name)

    os.makedirs(zip_extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(zip_extract_dir)

    # Find the smr.60s root folder inside this zip's extraction folder
    smr_root = os.path.join(zip_extract_dir, "home", "ezie", "smr.60s")

    if not os.path.isdir(smr_root):
        raise FileNotFoundError(
            f"Expected smr.60s folder not found after extraction: {smr_root}"
        )

    # Walk through every date subfolder found under smr_root
    days = {}
    for entry in sorted(os.listdir(smr_root)):
        if re.match(r"^\d{8}$", entry):
            day_folder = os.path.join(smr_root, entry)
            if os.path.isdir(day_folder):
                smr_files = []
                for f in os.listdir(day_folder):
                    if f.endswith('.smr.60s.txt'):
                        smr_files.append(os.path.join(day_folder, f))
                if smr_files:
                    days[entry] = sorted(smr_files)

    if not days:
        raise FileNotFoundError(
            f"No .smr.60s.txt files found under {smr_root} after extraction."
        )

    return days


def find_zip_for_date(date_str, raw_dir):
    """
    Find the daily zip file in raw_dir whose filename contains the
    given date.

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
        Full path to the matching daily zip file.
    ---------------------------------------------------------------------
    """

    import datetime

    # Validate format
    if not (date_str.isdigit() and len(date_str) == 8):
        raise ValueError(
            f"Invalid date '{date_str}'. Please provide exactly 8 digits in YYYYMMDD format."
        )

    # Validate it's a real calendar date
    try:
        datetime.datetime.strptime(date_str, "%Y%m%d")
    except ValueError:
        raise ValueError(
            f"'{date_str}' is not a valid calendar date."
        )

    # Search for matching zip - exclude weekly zips (which contain a
    # hyphen between two dates in their filename) to avoid false matches
    # when a date in the weekly range matches the daily date string.
    all_matches = glob.glob(os.path.join(raw_dir, f"*{date_str}*.zip"))
    matches = [m for m in all_matches if not re.search(r"\d{8}-\d{8}", os.path.basename(m))]

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No daily zip file found for date {date_str} in {raw_dir}/"
        )
    elif len(matches) > 1:
        raise ValueError(
            f"Multiple daily zip files match date {date_str}: {matches}"
        )

    return matches[0]


def find_zip_for_week(start_date_str, raw_dir):
    """
    Find the weekly zip file in raw_dir whose filename contains a date
    range starting with the given date.

    Weekly zip filenames follow the pattern:
        eziemag_YYYYMMDD-YYYYMMDD_<fingerprint>.zip
    where the first date is the start of the week.

    Parameters
    ---------------------------------------------------------------------
    start_date_str : str
        Start date of the week in YYYYMMDD format, e.g. "20260622".
    raw_dir : str
        Folder containing the raw zip files (e.g. "data/raw").
    ---------------------------------------------------------------------

    Returns
    ---------------------------------------------------------------------
    str
        Full path to the matching weekly zip file.
    ---------------------------------------------------------------------
    """

    import datetime

    # Validate format
    if not (start_date_str.isdigit() and len(start_date_str) == 8):
        raise ValueError(
            f"Invalid date '{start_date_str}'. Please provide exactly 8 digits in YYYYMMDD format."
        )

    try:
        datetime.datetime.strptime(start_date_str, "%Y%m%d")
    except ValueError:
        raise ValueError(
            f"'{start_date_str}' is not a valid calendar date."
        )

    # Search for weekly zips - these contain a hyphenated date range
    # (e.g. 20260622-20260628) and the start date must match
    all_zips = glob.glob(os.path.join(raw_dir, "*.zip"))
    matches = [
        z for z in all_zips
        if re.search(rf"{start_date_str}-\d{{8}}", os.path.basename(z))
    ]

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No weekly zip file found starting {start_date_str} in {raw_dir}/"
        )
    elif len(matches) > 1:
        raise ValueError(
            f"Multiple weekly zip files match start date {start_date_str}: {matches}"
        )

    return matches[0]


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
        also accepted for convenience.
    ---------------------------------------------------------------------

    Returns
    ---------------------------------------------------------------------
    pandas.DataFrame
        Combined DataFrame with all 24 EZIE-Mag columns, indexed by
        timestamp (timeString), sorted chronologically.
    ---------------------------------------------------------------------
    """

    # Convenience: wrap a single string in a list
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    dfs = []
    for path in file_paths:
        df = pd.read_csv(
            path,
            sep=r"\s+",
            engine="python",
            header=None
        )
        df.columns = SMR_COLUMNS
        dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)

    # Parse timestamps and set as index
    full_df['timeString'] = pd.to_datetime(full_df['timeString'])
    full_df = full_df.set_index('timeString').sort_index()

    return full_df


def round_and_fill_gaps(df):
    """
    Round timestamps to the nearest minute and reindex the DataFrame
    onto a complete 1-minute grid spanning the full calendar day.

    Background
    -----------
    Raw timestamps have sub-second precision that drifts slightly
    minute to minute due to small clock/sampling jitter in the device.
    This means real timestamps don't land on perfectly even 60-second
    boundaries, so a naive reindex would mark almost everything as
    missing.

    This function fixes both issues:
      1. Rounds each timestamp to its nearest whole minute
      2. Builds a complete 1-minute-interval index covering the full
         calendar day (00:00 to 23:59) and reindexes onto it - any
         minute with no real reading becomes a row of NaN, which
         matplotlib renders as a visible break in line plots

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

    df = df.copy()

    # Round to nearest minute to absorb sub-second jitter
    df.index = df.index.round("1min")

    # Guard against duplicate timestamps after rounding
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="first")]

    # Build a complete 1-minute grid for the whole day
    day_start = df.index.min().floor("D")
    full_day_index = pd.date_range(
        start=day_start,
        end=day_start + pd.Timedelta(hours=23, minutes=59),
        freq="1min"
    )

    # Reindex - missing minutes become NaN rows
    df = df.reindex(full_day_index)
    df.index.name = "timeString"

    return df


def add_derived_fields(df):
    """
    Add derived magnetic field features to a DataFrame that already has
    Bx, By, Bz columns.

    Adds:
      - B_total : total field magnitude, sqrt(Bx^2 + By^2 + Bz^2)
                  Comparable to observatory F value.
      - Bh      : horizontal field magnitude, sqrt(Bx^2 + By^2)
                  Comparable to observatory H value.
      - D       : declination in degrees, atan2(By, Bx)
                  Comparable to observatory D value.

    NaN rows (from gap-filling) produce NaN in all derived fields,
    consistent with the raw columns.

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

    df["B_total"] = np.sqrt(df["Bx"]**2 + df["By"]**2 + df["Bz"]**2)
    df["Bh"] = np.sqrt(df["Bx"]**2 + df["By"]**2)
    df["D"] = np.degrees(np.arctan2(df["By"], df["Bx"]))

    return df


def save_processed_day(df, date_str, fingerprint, processed_dir, overwrite=False):
    """
    Save a processed (gap-filled, feature-engineered) daily DataFrame
    to a fingerprint-based subfolder in processed_dir as a CSV.

    Output path:
        processed_dir/<fingerprint>/eziemag_<date_str>_<fingerprint>_processed.csv

    Does NOT silently overwrite an existing file - if the file already
    exists and overwrite=False, returns status "exists" without writing,
    so the calling notebook can decide whether to prompt the user.

    Parameters
    ---------------------------------------------------------------------
    df : pandas.DataFrame
        The processed DataFrame to save, indexed by timestamp.
    date_str : str
        Date in YYYYMMDD format, used to build the output filename.
    fingerprint : str
        The 12-character device fingerprint (e.g. "AAAAAIzs1uUA").
        Used as both the subfolder name and part of the filename.
    processed_dir : str
        Root folder for processed CSVs (e.g. "data/processed").
    overwrite : bool, default False
        If True, overwrite an existing file without being asked.
    ---------------------------------------------------------------------

    Returns
    ---------------------------------------------------------------------
    dict
        {
            "status": "saved" | "exists",
            "path": str   # full path to the output file
        }
    ---------------------------------------------------------------------
    """

    # Create the fingerprint subfolder if it doesn't exist
    device_dir = os.path.join(processed_dir, fingerprint)
    os.makedirs(device_dir, exist_ok=True)

    output_path = os.path.join(
        device_dir,
        f"eziemag_{date_str}_{fingerprint}_processed.csv"
    )

    if os.path.exists(output_path) and not overwrite:
        return {"status": "exists", "path": output_path}

    df.to_csv(output_path)

    return {"status": "saved", "path": output_path}


def load_processed_days(date_list, fingerprint, processed_dir):
    """
    Load and concatenate previously-saved processed daily CSVs for a
    specific device fingerprint.

    Reads from:
        processed_dir/<fingerprint>/eziemag_<date>_<fingerprint>_processed.csv

    Missing files (dates not yet processed) are skipped with a printed
    warning rather than raising an error, so a week or month with a few
    gaps still loads successfully.

    Parameters
    ---------------------------------------------------------------------
    date_list : list[str]
        List of YYYYMMDD date strings to load.
    fingerprint : str
        The 12-character device fingerprint to load data for.
    processed_dir : str
        Root folder for processed CSVs (e.g. "data/processed").
    ---------------------------------------------------------------------

    Returns
    ---------------------------------------------------------------------
    pandas.DataFrame
        Combined DataFrame, sorted by time, spanning all successfully
        loaded days. Returns an empty DataFrame if no files were found.
    ---------------------------------------------------------------------
    """

    dfs = []
    device_dir = os.path.join(processed_dir, fingerprint)

    for date_str in date_list:
        path = os.path.join(
            device_dir,
            f"eziemag_{date_str}_{fingerprint}_processed.csv"
        )

        if not os.path.exists(path):
            print(f"Warning: no processed file found for {date_str} ({fingerprint}), skipping.")
            continue

        day_df = pd.read_csv(path, index_col="timeString", parse_dates=True)
        dfs.append(day_df)

    if len(dfs) == 0:
        print(f"Warning: no processed files found for any requested date ({fingerprint}).")
        return pd.DataFrame()

    combined = pd.concat(dfs)
    return combined.sort_index()


def list_available_fingerprints(processed_dir):
    """
    List all device fingerprints that have processed data available,
    by scanning the subfolders of processed_dir.

    Useful in the Daily/Weekly notebooks so you don't have to remember
    fingerprint strings - just call this to see what's available.

    Parameters
    ---------------------------------------------------------------------
    processed_dir : str
        Root folder for processed CSVs (e.g. "data/processed").
    ---------------------------------------------------------------------

    Returns
    ---------------------------------------------------------------------
    list[str]
        List of fingerprint strings (subfolder names) found in
        processed_dir, sorted alphabetically.
    ---------------------------------------------------------------------
    """

    if not os.path.isdir(processed_dir):
        return []

    fingerprints = []
    for entry in sorted(os.listdir(processed_dir)):
        if os.path.isdir(os.path.join(processed_dir, entry)):
            name = STATION_NAMES.get(entry, "Unknown device")
            fingerprints.append((entry, name))

    return fingerprints