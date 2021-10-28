import fire as fire
import pandas as pd

from utils_crunchy import normalize_crunchbase_df

CB_ANN_DATE = 'Announced Date'
CB_LEAD_INV = 'Lead Investors'
CB_ORG_UID = 'Organization Name URL'
ADD_LEAD_COL = 'Lead'


def first_investment_rounds(cb_csv, check_lead='Tiger Global Management', out=None):
    if cb_csv is None:
        raise ValueError("Please provide a csv file")

    # load file
    print(f'\nLoading {cb_csv}...')

    df_orig = pd.read_csv(cb_csv)
    df_date_desc = df_orig.sort_values(by=CB_ANN_DATE, ascending=False)
    df_unique_first_round = df_date_desc.drop_duplicates(subset=CB_ORG_UID, keep='last')
    df_out = df_unique_first_round.copy()
    if check_lead is not None:
        df_out[ADD_LEAD_COL] = df_out.apply(lambda row: '?' if pd.isna(row[CB_LEAD_INV]) else '1' if check_lead in row[CB_LEAD_INV] else '0', axis=1)
    df_out, df_out_headers = normalize_crunchbase_df(df_out)

    # print the number of rows for the original and the filtered dataframes
    print(f'\nOriginal: {len(df_orig)} rows, date-descending {len(df_date_desc)}, unique {len(df_unique_first_round)}, output {len(df_out)}')

    # write the output file
    if out is not None:
        print(f'\nSaving to {out}...')
        df_out.to_csv(out, index=False, columns=df_out_headers)
    else:
        print(f'\nWARNING: output file not specified')


if __name__ == '__main__':
    fire.Fire(first_investment_rounds)
