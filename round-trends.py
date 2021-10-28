import fire as fire
import pandas as pd

from utils_crunchy import normalize_crunchbase_df

CB_ANN_DATE = 'Announced Date'
CB_ORG_UID = 'Organization Name URL'
CB_LEAD_INV = 'Lead Investors'
ADD_LEAD_COL = 'Lead'


def remove_follow_ons(df_orig):
    df_date_desc = df_orig.sort_values(by=CB_ANN_DATE, ascending=False)
    df_no_follow_ons = df_date_desc.drop_duplicates(subset=CB_ORG_UID, keep='last')
    print(f'\nOriginal: {len(df_orig)} rows, date-descending {len(df_date_desc)}, unique {len(df_no_follow_ons)}')
    return df_no_follow_ons.copy()


def add_lead_col(df, check_lead):
    df[ADD_LEAD_COL] = df.apply(lambda row: '?' if pd.isna(row[CB_LEAD_INV]) else '1' if check_lead in row[CB_LEAD_INV] else '0', axis=1)


def main(cb_csv, check_lead='Tiger Global Management', out=None):
    if cb_csv is None:
        raise ValueError("Please provide a csv file")

    # Load, remove follow-ons, add lead column
    print(f'\nLoading {cb_csv}...')
    df = remove_follow_ons(pd.read_csv(cb_csv))
    if check_lead is not None:
        add_lead_col(df, check_lead)
    df, df_out_headers = normalize_crunchbase_df(df)

    # Save to file
    if out is None:
        print(f'\nWARNING: output file not specified')
        return
    print(f'\nSaving to {out}...')
    df.to_csv(out, index=False, columns=df_out_headers)


if __name__ == '__main__':
    fire.Fire(main)
