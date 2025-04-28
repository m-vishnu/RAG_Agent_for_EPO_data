import pandas as pd

def data_loader(excel_file_path) -> pd.DataFrame:
    """
    Load the data from an Excel file and return a DataFrame.
    """
    boa_df = pd.read_excel(excel_file_path)

    return boa_df

def data_preprocessor(boa_df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with NaN values in the 'Title of Invention' column
    """

    boa_df_proc = boa_df[~boa_df['Title of Invention'].isna()]
    return boa_df_proc

def get_data(excel_file_path) -> pd.DataFrame:
    """
    Load and preprocess the data.
    """
    boa_df = data_loader(excel_file_path)
    boa_df_preproc = data_preprocessor(boa_df)
    return boa_df_preproc

if __name__ == "__main__":
    pass
