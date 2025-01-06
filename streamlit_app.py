import streamlit as st
import pandas as pd
import re
from thefuzz import fuzz
import datetime

# --------------- PART 1: SAGE CLEANING LOGIC (from your "Sage Export Cleaner") ---------------

def process_card_data(df, rows_to_delete_top=0, rows_to_delete_bottom=0, card_type="Capital One"):
    """Processes Sage export data in a pandas DataFrame, removing top/bottom rows,
       finding the 'Verified' header row, renaming columns, extracting 'Card No.', etc."""
    try:
        # 1) Remove top N rows
        df = df.iloc[rows_to_delete_top:].reset_index(drop=True)

        # 2) Find the row with 'Verified' within first 10 lines
        verified_header_index = None
        for i in range(min(10, len(df))):
            row_vals = df.iloc[i].astype(str).values
            if any("Verified" in cell for cell in row_vals):
                verified_header_index = i
                break

        if verified_header_index is None:
            raise ValueError("Could not find a row containing 'Verified' within the first 10 rows.")

        # 3) Set that row as the header
        df.columns = df.iloc[verified_header_index].values
        df = df.iloc[verified_header_index + 1:].reset_index(drop=True)

        # 4) Clean column names (convert NaN to "", strip spaces)
        cleaned_cols = []
        for col in df.columns:
            if pd.isnull(col):
                cleaned_cols.append("")
            else:
                cleaned_cols.append(str(col).strip())
        df.columns = cleaned_cols

        # 5) Rename all "" columns to CardHolderName1, CardHolderName2, etc.
        count_empty = 0
        for i, col in enumerate(df.columns):
            if col == "":
                count_empty += 1
                df.columns.values[i] = f"CardHolderName{count_empty}"

        # 6) Drop duplicate columns (keeping the first)
        df = df.loc[:, ~df.columns.duplicated()]

        # 7) Drop "Record#" if it exists
        if "Record#" in df.columns:
            df.drop(columns=["Record#"], inplace=True)

        # 8) Identify and rename CardHolderName column
        card_cols = [c for c in df.columns if c.startswith("CardHolderName")]
        chosen_col = None
        for c in card_cols:
            sample_values = df[c].dropna().astype(str).head(20)
            if any(re.search(r"\d+", val) for val in sample_values):
                chosen_col = c
                break
        if chosen_col is not None:
            df.rename(columns={chosen_col: "CardHolderName"}, inplace=True)
            for c in card_cols:
                if c != chosen_col:
                    df.drop(columns=[c], inplace=True, errors="ignore")

        # 9) Extract digits from "CardHolderName", forward-fill
        if "CardHolderName" in df.columns:
            df["Card No."] = df["CardHolderName"].astype(str).str.extract(r"(\d+)").ffill()
            df.drop(columns=["CardHolderName"], inplace=True)

        # 10) Keep only the final columns
        desired_cols = ["Card No.", "Verified", "Date", "Payee", "Credits", "Charges"]
        existing_cols = [c for c in desired_cols if c in df.columns]
        df = df[existing_cols]

        # 11) Drop rows that are empty in Verified/Date/Payee/Charges (besides Card No.)
        check_cols = [c for c in existing_cols if c != "Card No."]
        df = df.dropna(subset=check_cols, how="all")

        # 12) Remove the last N rows
        if rows_to_delete_bottom > 0 and len(df) >= rows_to_delete_bottom:
            df = df.iloc[:-rows_to_delete_bottom]

        # Additional processing based on card_type
        if card_type == "Amex":
            # Apply any specific cleaning steps for Amex Sage exports if needed
            pass  # Replace with actual logic if needed

        return df
    except Exception as e:
        st.error(f"Error during Sage data processing: {e}")
        return None  # Return None in case of error

def download_csv(df, filename="cleaned_card_data.csv"):
    """Utility to let user download a pandas DataFrame as a CSV file."""
    if df is not None and not df.empty:
        csv_string = df.to_csv(index=False, encoding='utf-8')
        st.download_button(
            label="Download CSV",
            data=csv_string,
            file_name=filename,
            mime="text/csv",
        )

# --------------- PART 2: CREDIT CARD CLEANING LOGIC ---------------

def clean_credit_card_data(df_cc: pd.DataFrame, card_type: str) -> pd.DataFrame:
    """
    Cleans credit card data based on the card type (Capital One or Amex).

    Args:
        df_cc (pd.DataFrame): Raw credit card data.
        card_type (str): Type of the credit card ('Capital One' or 'Amex').

    Returns:
        pd.DataFrame: Cleaned credit card data.
    """
    if card_type == "Capital One":
        # Define renaming map for Capital One
        rename_map = {
            "Transaction Date": "CC_Transaction_Date",
            "Posted Date": "CC_Posted_Date",
            "Card No.": "CC_Card_No",
            "Description": "CC_Description",
            "Category": "CC_Category",
            "Debit": "CC_Debit",
        }
    elif card_type == "Amex":
        # Define renaming map for Amex
        rename_map = {
            "Date": "CC_Transaction_Date",
            "Amount": "CC_Debit",
            "Account #": "CC_Card_No",
            "Description": "CC_Description",
            # Add other necessary mappings if headers differ
        }
    else:
        st.error("Unsupported card type selected.")
        return pd.DataFrame()  # Return empty DataFrame if unsupported

    # Rename columns
    df_cc.rename(columns=rename_map, inplace=True)

    if card_type == "Amex":
        # Remove "Receipt" and "Card Member" columns if they exist
        columns_to_remove = ["Receipt", "Card Member"]
        for col in columns_to_remove:
            if col in df_cc.columns:
                df_cc.drop(columns=[col], inplace=True)

        # Remove '-' from 'CC_Card_No'
        if "CC_Card_No" in df_cc.columns:
            df_cc["CC_Card_No"] = df_cc["CC_Card_No"].astype(str).str.replace("-", "", regex=False)

    # Common cleaning steps for both card types
    # Convert dates
    if "CC_Transaction_Date" in df_cc.columns:
        df_cc["CC_Transaction_Date"] = pd.to_datetime(df_cc["CC_Transaction_Date"], errors="coerce")
    if "CC_Posted_Date" in df_cc.columns:
        df_cc["CC_Posted_Date"] = pd.to_datetime(df_cc["CC_Posted_Date"], errors="coerce")

    # Convert numeric
    if "CC_Debit" in df_cc.columns:
        df_cc["CC_Debit"] = pd.to_numeric(df_cc["CC_Debit"], errors="coerce")

    # Ensure card number is string
    if "CC_Card_No" in df_cc.columns:
        df_cc["CC_Card_No"] = df_cc["CC_Card_No"].astype(str)

    return df_cc

# --------------- PART 3: DATE MATCHING LOGIC (±3 days) ---------------

def within_3_days(sage_date, cc_trans_date, cc_posted_date):
    """
    Returns True if 'sage_date' is within 3 days of EITHER
    cc_trans_date OR cc_posted_date.
    """
    if pd.isnull(sage_date):
        return False

    # Check difference with Transaction Date
    within_trans = False
    if not pd.isnull(cc_trans_date):
        diff_trans = abs((sage_date - cc_trans_date).days)
        within_trans = (diff_trans <= 3)

    # Check difference with Posted Date
    within_posted = False
    if not pd.isnull(cc_posted_date):
        diff_posted = abs((sage_date - cc_posted_date).days)
        within_posted = (diff_posted <= 3)

    return (within_trans or within_posted)

# --------------- FIX: Ensure numeric consistency for amounts during comparison ---------------

def ensure_numeric(column):
    """Converts a column to numeric, removing any quotation marks or invalid characters."""
    return pd.to_numeric(column.replace(r'["\',]', '', regex=True), errors='coerce')

# Updated find_discrepancies_3day function

def find_discrepancies_3day(df_cc: pd.DataFrame, df_sage: pd.DataFrame) -> pd.DataFrame:
    """
    Matching logic with a 3-day window:
     - Card No. must match exactly
     - Amount must match exactly (ensure numeric consistency)
     - Sage date must be within +/- 3 days of either CC_Transaction_Date or CC_Posted_Date
    If no match found, we list it as "No Match in Sage".
    """
    # Make sure Sage columns match expected names
    rename_sage = {
        "Date": "Sage_Date",
        "Payee": "Sage_Payee",
        "Card No.": "Sage_Card_No",
        "Charges": "Sage_Charges",
        "Verified": "Sage_Verified",
    }
    df_sage.rename(columns=rename_sage, inplace=True, errors="ignore")

    # Convert Sage dates to datetime
    if "Sage_Date" in df_sage.columns:
        df_sage["Sage_Date"] = pd.to_datetime(df_sage["Sage_Date"], errors="coerce")

    # Convert amounts, card no, and ensure numeric consistency
    if "Sage_Charges" in df_sage.columns:
        df_sage["Sage_Charges"] = ensure_numeric(df_sage["Sage_Charges"])
    if "Sage_Card_No" in df_sage.columns:
        df_sage["Sage_Card_No"] = df_sage["Sage_Card_No"].astype(str)

    # Apply the same cleanup for credit card data
    if "CC_Debit" in df_cc.columns:
        df_cc["CC_Debit"] = ensure_numeric(df_cc["CC_Debit"])

    mismatch_rows = []

    # For each CC row
    for idx, row in df_cc.iterrows():
        cc_trans_date = row.get("CC_Transaction_Date")
        cc_posted_date = row.get("CC_Posted_Date")
        cc_card_no = row.get("CC_Card_No")
        cc_amount = row.get("CC_Debit")
        cc_desc = row.get("CC_Description") if "CC_Description" in row else None

        # Filter Sage by card no. and amount first
        potential_matches = df_sage[
            (df_sage["Sage_Card_No"] == str(cc_card_no)) &
            (df_sage["Sage_Charges"] == cc_amount)
        ].copy()

        # Further filter by date within 3 days
        potential_matches = potential_matches[
            potential_matches["Sage_Date"].apply(lambda sd: within_3_days(sd, cc_trans_date, cc_posted_date))
        ]

        if len(potential_matches) == 0:
            # No match found
            mismatch_rows.append({
                "CC_Transaction_Date": cc_trans_date,
                "CC_Posted_Date": cc_posted_date,
                "CC_Card_No": cc_card_no,
                "CC_Debit": cc_amount,
                "CC_Description": cc_desc,
                "Sage_Date": None,
                "Sage_Payee": None,
                "Sage_Charges": None,
                "Status": "No Match in Sage (3-day window)"
            })
        else:
            # Just pick the first match for demonstration
            best_match = potential_matches.iloc[0]
            mismatch_rows.append({
                "CC_Transaction_Date": cc_trans_date,
                "CC_Posted_Date": cc_posted_date,
                "CC_Card_No": cc_card_no,
                "CC_Debit": cc_amount,
                "CC_Description": cc_desc,
                "Sage_Date": best_match["Sage_Date"],
                "Sage_Payee": best_match.get("Sage_Payee"),
                "Sage_Charges": best_match["Sage_Charges"],
                "Status": "Matched (within 3-day window)"
            })

    return pd.DataFrame(mismatch_rows)

# --------------- PART 4: STREAMLIT APP ---------------

def main():
    st.set_page_config(page_title="Sage AutoReconcile", page_icon=":credit_card:", layout="wide")
    st.title("Sage AutoReconcile (±3-Day Matching)")

    st.markdown(
    """
    <style>
    
    /* Center-align all dataframes */
    .stDataFrame {
        margin-left: auto;
        margin-right: auto;
        text-align: center;
    }
    
    .stButton > button {
        background-color: #4CAF50; /* Green */
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;
    }

    .stButton > button:hover {
        background-color: white;
    }

    h1, h2, h3 {
        font-family: 'Arial', sans-serif;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }

    h1 {
        font-size: 36px;
        font-weight: bold;
    }

    h2 {
        font-size: 28px;
    }

    h3 {
        font-size: 22px;
    }


    </style>
    """, unsafe_allow_html=True
    )

    
    st.markdown("---")  # Horizontal line

    st.write("""
    This app:
    1. **Cleans Sage Export Data:** Processes and standardizes Sage export files for accurate reconciliation.
    
    2. **Cleans Credit Card Statements:** Formats and prepares credit card statement data for comparison.
    
    3. **Compares Transactions:** Matches transactions allowing for a ±3-day window between either the 
       transaction date or posted date (Credit Card) and the Sage date.
       
    4. **Generates Reconciliation Reports:** Provides a downloadable CSV file detailing matched and unmatched transactions. 
    """)

    st.markdown("---")  # Horizontal line

    # Layout improvements: Columns for better organization
    col1, col2 = st.columns(2)

    # Upload Sage
    with col1:
        st.header("Upload Sage Export (Raw CSV) 🗃️")
        sage_file = st.file_uploader("Upload Sage CSV (no header in the file)", type=["csv"])
        rows_top = st.number_input("Rows to remove from top of Sage", min_value=0, value=0, help="Enter the number of rows to skip at the top of your Sage file.")
        rows_bottom = st.number_input("Rows to remove from bottom of Sage", min_value=0, value=0, help="Enter the number of rows to delete at the bottom of your Sage file.")
    
    st.markdown("---")  # Horizontal line
    
    # Upload CC
    with col2:
        st.header("Upload Credit Card CSV 💳")
        card_type = st.selectbox(
            "Select Credit Card Type",
            options=["Capital One", "Amex"],
            help="Choose the type of credit card statement you are uploading."
        )
        cc_file = st.file_uploader("Upload CC CSV (with normal headers)", type=["csv"])

    if sage_file and cc_file:
        if st.button("Process & Compare"):
            try:
                # Read the Sage file with minimal assumptions (no header)
                df_sage_raw = pd.read_csv(sage_file, header=None)
                # Clean it with process_card_data
                df_sage_cleaned = process_card_data(
                    df_sage_raw,
                    rows_to_delete_top=rows_top,
                    rows_to_delete_bottom=rows_bottom,
                    card_type=card_type  # Pass card_type if needed
                )

                if df_sage_cleaned is None or df_sage_cleaned.empty:
                    st.error("Sage data is empty or could not be processed.")
                    st.stop()

                st.subheader("Cleaned Sage Data Preview")
                st.dataframe(df_sage_cleaned)

                # Read & clean CC file
                df_cc_raw = pd.read_csv(cc_file)
                df_cc_cleaned = clean_credit_card_data(df_cc_raw, card_type)

                if df_cc_cleaned.empty:
                    st.error("Credit card data could not be processed.")
                    st.stop()

                st.subheader("Cleaned Credit Card Data Preview")
                st.dataframe(df_cc_cleaned)

                # Compare with 3-day logic
                st.write("## Discrepancy Report (±3-day Window)")
                results_df = find_discrepancies_3day(df_cc_cleaned, df_sage_cleaned)
                st.dataframe(results_df)

                # Download the mismatch
                csv_data = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Discrepancies CSV",
                    data=csv_data,
                    file_name="cc_sage_discrepancies.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"An error occurred: {e}")

    else:
        st.info("Please upload both Sage and Credit Card files to proceed.")

if __name__ == "__main__":
    main()