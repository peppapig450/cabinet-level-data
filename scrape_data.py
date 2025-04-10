import os
import re

import pandas as pd
from pandas.api.types import is_float_dtype, is_object_dtype


def create_cabinet_csv(
    url: str, administration: str, output_file: str | None = None
) -> pd.DataFrame:
    """
    Create a CSV database of a presidential cabinet from a Wikipedia page using vectorized operations.

    Args:
        url (str): URL of the Wikipedia page
        administration (str): Name of the administration (e.g., "Trump 1st", "Biden")
        output_file (str, optional): Path to save the CSV file. If None, will use a default name.

    Returns:
        pandas.DataFrame: The processed cabinet data
    """
    # Extract cabinet name from URL for default filename
    cabinet_name = url.split("/")[-1].replace("_", " ")
    if output_file is None:
        output_file = f"{cabinet_name.lower().replace(' ', '_')}.csv"

    # Read all tables containing "Date of birth"
    print(f"Fetching tables from {cabinet_name}...")
    df_list = pd.read_html(url, match="Date of birth")
    print(f"Found {len(df_list)} tables.")

    processed_dfs: list[pd.DataFrame] = []

    # Process each dataframe
    for i, df in enumerate(df_list):
        print(f"Processing table {i + 1}...")

        try:
            # Skip tables without MultiIndex
            if not isinstance(df.columns, pd.MultiIndex):
                print(f"Skipping table {i + 1} as it doesn't have a MultiIndex.")
                continue

            # Get the position from the first level of MultiIndex
            position = df.columns[0][0]

            # Create flattened column names
            if isinstance(df.columns, pd.MultiIndex):
                # Extract the position name from the first level
                position = df.columns.get_level_values(0)[0]

                level0 = df.columns.get_level_values(0)
                level1 = (
                    df.columns.get_level_values(1) if df.columns.nlevels > 1 else None
                )

                if level1 is not None:
                    mask = level1.astype(str).str.strip() != ""
                    new_cols = pd.Series(level1.where(mask, level0))
                else:
                    new_cols = pd.Series(level0)

                if (duplicated := new_cols.duplicated(keep=False)).any():
                    for col in new_cols[duplicated].unique():
                        indices = new_cols[new_cols == col].index

                        for suffix_idx, idx in enumerate(indices[1:], 1):
                            new_cols.iloc[idx] = f"{col}_{suffix_idx}"

                df.columns = new_cols

            # Drop unnamed columns
            unnamed_pattern = "Unnamed|^$"
            df = df.loc[:, ~df.columns.astype(str).str.contains(unnamed_pattern)]

            # Add position column
            df["Position"] = position

            df["Administration"] = administration

            if "Name" in df.columns:
                # Filter out rows where Name is NA or equals 'Name' (header rows)
                df = df[df["Name"].notna() & (df["Name"] != "Name")]

            # Clean potential HTML tags or extra whitespace
            if "Name" in df.columns and is_object_dtype(df["Name"]):
                # Remove HTML tags
                df["Name"] = df["Name"].str.replace(r"<[^>]+>", "", regex=True)
                # Normalize whitespace
                df["Name"] = df["Name"].str.replace(r"\s+", " ", regex=True).str.strip()

            processed_dfs.append(df)

        except Exception as e:
            print(f"Error processing table {i + 1}: {str(e)}")

    # Concatenate all processed dataframes
    if not processed_dfs:
        print("No tables were processed successfully.")
        return pd.DataFrame()

    final_df = pd.concat(processed_dfs, ignore_index=True)

    # Handle null values consistently
    final_df = final_df.replace(["", "N/A"], pd.NA)

    # Process data part
    if "Date of birth" in final_df.columns:
        dob_col = final_df["Date of birth"]

        # Extract just the date part (before the parentheses)
        date_part = dob_col.str.extract(r"(.*?)\s*\(", expand=False).fillna(dob_col)

        final_df["Birth Date"] = pd.to_datetime(date_part, errors="coerce")

        # Extract birth year as integer
        final_df["Birth Year"] = final_df["Birth Date"].dt.year

        # Handle cases where datetime conversion failed
        if final_df["Birth Year"].isna().any():
            # Fallback to regex extraction for problematic rows
            year_mask = final_df["Birth Year"].isna()

            year_extracted = dob_col[year_mask].str.extract(r"(\d{4})", expand=False)
            final_df.loc[year_mask, "Birth Year"] = pd.to_numeric(
                year_extracted, errors="coerce", downcast="unsigned"
            )

        # Extract age
        final_df["Age"] = dob_col.str.extract(r"(?:age|Age|AGE)\s*(\d+)", expand=False)
        final_df["Age"] = pd.to_numeric(
            final_df["Age"], errors="coerce", downcast="integer"
        )

    # Process 'Years' column if it exists
    if "Years" in final_df.columns:
        # More flexible pattern to match various formats
        year_pattern = r"(\d{4})(?:\s*[–—-]\s*(?:present|(\d{4})))?"
        year_extracted = final_df["Years"].str.extract(year_pattern)

        # Convert to numeric types
        final_df["Start year"] = pd.to_numeric(year_extracted[0], errors="coerce")
        final_df["End year"] = pd.to_numeric(year_extracted[1], errors="coerce")

        # Create boolean column for current positions
        final_df["Current"] = final_df["Years"].str.contains(
            "present", case=False, na=False
        )

    # Extract state information from background if state column doesn't exist
    if "State" not in final_df.columns and "Background" in final_df.columns:
        # Look for state patterns in background
        state_pattern = (
            r"(?:senator|representative|governor)(?:\s+from|\s+of)\s+([A-Za-z\s]+)"
        )
        final_df["State"] = final_df["Background"].str.extract(
            state_pattern, flags=re.IGNORECASE
        )[0]

    # Drop empty columns
    final_df = final_df.dropna(axis=1, how="all")

    # Drop redundant columns
    columns_to_drop = ["Reference", "Refs", "Notes"]
    cols_to_remove = final_df.columns.intersection(columns_to_drop)
    if not cols_to_remove.empty:
        final_df = final_df.drop(columns=cols_to_remove)

    # Reorder columns with key information first
    priority_cols = [
        "Administration",
        "Position",
        "Name",
        "Age",
        "State",
        "Date of birth",
        "Birth Year",
        "Background",
    ]
    available_priority = final_df.columns.intersection(priority_cols)
    remaining_cols = final_df.columns.difference(priority_cols)
    # Preserve priority order by reindexing available columns
    available_priority_ordered = pd.Index(
        [col for col in priority_cols if col in available_priority]
    )
    final_df = final_df[available_priority_ordered.append(remaining_cols)]

    # Convert datetime to ISO format when saving to CSV
    if "Birth Date" in final_df.columns:
        final_df["Birth Date"] = final_df["Birth Date"].dt.strftime("%Y-%m-%d")

    # Convert floating point values to integers where appropriate
    for col in ["Birth Year", "Start year", "End year"]:
        if col in final_df.columns and is_float_dtype(final_df[col]):
            final_df[col] = final_df[col].fillna(pd.NA).astype("Int64")

    os.makedirs("data", exist_ok=True)

    # Save to CSV with proper handling of newlines in text fields
    final_df.to_csv(
        f"data/{output_file}", index=False, quoting=1, escapechar="\\", doublequote=True
    )
    print(f"CSV file saved to {os.path.abspath(output_file)}")

    return final_df


def combine_cabinets(csv_files, output_file="all_cabinets.csv"):
    """
    Combine multiple cabinet CSV files into a single file using pandas.

    Args:
        csv_files (list): List of CSV files to combine
        output_file (str): Name of the output file

    Returns:
        pandas.DataFrame: Combined cabinet data
    """
    print(f"Combining {len(csv_files)} cabinet files...")

    # Using a list comprehension with pd.read_csv and error handling
    dfs = []
    for file in csv_files:
        try:
            # Use pandas read_csv with appropriate options for consistency
            df = pd.read_csv(
                f"data/{file}",
                na_values=["", "N/A"],
                keep_default_na=True,
                dtype_backend="numpy_nullable",  # Better handling of nullable integer types
            )
            dfs.append(df)
            print(
                f"Successfully read {file} with {len(df)} rows and {len(df.columns)} columns"
            )
        except Exception as e:
            print(f"Error reading {file}: {str(e)}")

    if not dfs:
        print("No CSV files were read successfully.")
        return pd.DataFrame()

    combined_df = pd.concat(dfs, ignore_index=True, sort=False)

    # Process the combined data for consistency
    print(f"Combined data has {len(combined_df)} total rows")

    # Ensure directory exists
    os.makedirs("data", exist_ok=True)

    # Save the combined data
    combined_df.to_csv(
        f"data/{output_file}",
        index=False,
        quoting=1,  # Quote all strings
        escapechar="\\",
        doublequote=True,
        na_rep="",  # Empty string for NA values
    )
    print(f"Combined CSV file saved to data/{output_file}")

    return combined_df


if __name__ == "__main__":
    # Define cabinet sources with their metadata
    cabinet_sources = [
        {
            "url": "https://en.wikipedia.org/wiki/Second_cabinet_of_Donald_Trump",
            "administration": "Trump 2nd",
            "output_file": "trump_second_cabinet.csv",
        },
        {
            "url": "https://en.wikipedia.org/wiki/First_cabinet_of_Donald_Trump",
            "administration": "Trump 1st",
            "output_file": "trump_first_cabinet.csv",
        },
        {
            "url": "https://en.wikipedia.org/wiki/Cabinet_of_Joe_Biden",
            "administration": "Biden",
            "output_file": "biden_cabinet.csv",
        },
    ]

    # Process each cabinet using pandas
    results = {}
    csv_files = []

    # Create individual cabinet files
    for cabinet in cabinet_sources:
        print(f"\nProcessing {cabinet['administration']} cabinet...")
        df = create_cabinet_csv(
            url=cabinet["url"],
            administration=cabinet["administration"],
            output_file=cabinet["output_file"],
        )
        results[cabinet["administration"]] = df
        csv_files.append(cabinet["output_file"])

    # Display summary of individual results
    print("\nSummary of cabinets processed:")
    for admin, df in results.items():
        print(f"{admin}: {len(df)} members")

    # Combine all cabinets into a single file
    print("\nCombining all cabinet data...")
    all_cabinets = combine_cabinets(csv_files)

    # Final summary
    print(
        f"\nAll cabinet data processing complete! Total of {len(all_cabinets)} cabinet members processed."
    )
