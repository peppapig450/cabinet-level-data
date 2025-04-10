import os
import re

import pandas as pd
from pandas.api.types import is_float_dtype, is_object_dtype


def create_cabinet_csv(url: str, output_file: str | None = None) -> pd.DataFrame:
    """
    Create a CSV database of a presidential cabinet from a Wikipedia page using vectorized operations.

    Args:
        url (str): URL of the Wikipedia page
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
                position = df.columns[0][0]

                # For each column tuple, prefer second level if it exists and isn't empty
                new_cols = pd.Series(df.columns.to_list()).apply(
                    lambda x: (
                        x[1]
                        if isinstance(x, tuple)
                        and len(x) > 1
                        and x[1]
                        and str(x[1]).strip() != ""
                        else (x[0] if isinstance(x, tuple) else x)
                    )
                )

                # Handle duplicate column names
                col_counts = new_cols.value_counts().to_dict()

                # Only add suffix to duplicates
                for col, count in col_counts.items():
                    if count > 1:
                        # Find all indices where this column name appears
                        indices = new_cols[new_cols == col].index
                        # Add suffixes to all but the first occurrence
                        for suffix_idx, idx in enumerate(indices[1:], 1):
                            new_cols.iloc[idx] = f"{col}_{suffix_idx}"

                # Assign the new column names to the dataframe
                df.columns = new_cols

            # Drop unnamed columns
            unnamed_pattern = "Unnamed|^$"
            df = df.loc[:, ~df.columns.astype(str).str.contains(unnamed_pattern)]

            # Add position column
            df["Position"] = position

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
        # Extract just the date part (before the parantheses)
        date_part = final_df["Date of birth"].str.extract(r"(.*?)\s*\(", expand=False)
        date_part = date_part.fillna(final_df["Date of birth"])

        # Convert to datetime preserving the original string column
        final_df["Birth Date"] = pd.to_datetime(date_part, errors="coerce")

        # Extract birth year as integer
        final_df["Birth Year"] = final_df["Birth Date"].dt.year

        # Handle cases where datetime conversion failed
        if final_df["Birth Year"].isna().any():
            # Fallback to regex extraction for problematic rows
            year_mask = final_df["Birth Year"].isna()
            year_extracted = final_df.loc[year_mask, "Date of birth"].str.extract(
                r"(\d{4})"
            )
            final_df.loc[year_mask, "Birth Year"] = pd.to_numeric(
                year_extracted[0], errors="coerce", downcast="unsigned"
            )

        # Extract age
        final_df["Age"] = final_df["Date of birth"].str.extract(
            r"(?:age|Age|AGE)\s*(\d+)", expand=False
        )
        final_df["Age"] = pd.to_numeric(
            final_df["Age"], errors="coerce", downcast="unsigned"
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
    final_df = final_df.drop(
        [col for col in columns_to_drop if col in final_df.columns], axis=1
    )

    # Reorder columns with key information first
    priority_cols = [
        "Position",
        "Name",
        "Age",
        "State",
        "Date of birth",
        "Birth Year",
        "Background",
    ]
    available_priority = [col for col in priority_cols if col in final_df.columns]
    remaining_cols = [col for col in final_df.columns if col not in priority_cols]
    final_df = final_df[available_priority + remaining_cols]

    # Convert datetime to ISO format when saving to CSV
    if "Birth Date" in final_df.columns:
        final_df["Birth Date"] = final_df["Birth Date"].dt.strftime("%Y-%m-%d")

    # Convert floating point values to integers where appropriate
    for col in ["Birth Year", "Start year", "End year"]:
        if col in final_df.columns and is_float_dtype(final_df[col]):
            final_df[col] = final_df[col].fillna(pd.NA).astype("Int64")

    # Save to CSV with proper handling of newlines in text fields
    final_df.to_csv(
        f"data/{output_file}", index=False, quoting=1, escapechar="\\", doublequote=True
    )
    print(f"CSV file saved to {os.path.abspath(output_file)}")

    return final_df


if __name__ == "__main__":
    create_cabinet_csv(
        "https://en.wikipedia.org/wiki/Second_cabinet_of_Donald_Trump",
        "trump_second_cabinet.csv",
    )
    create_cabinet_csv(
        "https://en.wikipedia.org/wiki/First_cabinet_of_Donald_Trump",
        "trump_first_cabinet.csv",
    )
    create_cabinet_csv(
        "https://en.wikipedia.org/wiki/Cabinet_of_Joe_Biden", "biden_cabinet.csv"
    )
