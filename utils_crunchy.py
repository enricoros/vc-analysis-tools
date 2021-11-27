import numpy as np

# uniform arrangement of the data frame
COL_NAME = 'Name'
COL_TITLE = 'Title'
COL_SERIES = 'Series'
COL_MONEY = 'Money'
COL_FUND_DATE = 'Funds Date'
COL_FUND_YEAR = 'Funds Year'
COL_LABEL = 'Label'
COL_DESCRIPTION = 'Description'
COL_INDUSTRIES = 'Industries'
_TSV_HEADERS = [COL_NAME, COL_TITLE, COL_SERIES, COL_MONEY, COL_FUND_DATE, COL_FUND_YEAR, COL_LABEL, COL_DESCRIPTION, COL_INDUSTRIES]
_TSV_OPTIONALS = ['Lead', 'Website']


# data loader: df[ Title, Name, Series, Money, Industries, Description ]
def normalize_crunchbase_df(df):
    # type heuristics: pass-through
    if "Label" in df:
        print(' * detected a pass-through CSV')

    # type heuristics: Funding Rounds
    elif "Money Raised Currency (in USD)" in df and "Organization Industries" in df:
        print(' * detected a Funding Rounds CSV')
        df.rename(columns={
            "Organization Name": COL_NAME,
            "Funding Type": COL_SERIES,
            "Announced Date": COL_FUND_DATE,
            "Money Raised Currency (in USD)": COL_MONEY,
            "Organization Industries": COL_INDUSTRIES,
            "Organization Description": COL_DESCRIPTION,
        }, inplace=True)
        if COL_MONEY in df:
            df[COL_MONEY] = df[COL_MONEY] / 1E+06

    # type heuristics: Company List
    elif "Total Funding Amount Currency (in USD)" in df:
        print(' * detected a Company List CSV')
        df.rename(columns={
            "Organization Name": COL_NAME,
            # Series
            # Funding Date
            "Total Funding Amount Currency (in USD)": COL_MONEY,
            "Industries": COL_INDUSTRIES,
            "Description": COL_DESCRIPTION,
        }, inplace=True)
        if 'Last Funding Type' in df:
            df.rename(columns={"Last Funding Type": COL_SERIES}, inplace=True)
        else:
            df[COL_SERIES] = 'Unknown'
        if 'Last Funding Date' in df:
            df.rename(columns={"Last Funding Date": COL_FUND_DATE}, inplace=True)
        else:
            df[COL_FUND_DATE] = 'Unknown'
        if COL_MONEY in df:
            df[COL_MONEY] = df[COL_MONEY] / 1E+06

    # type heuristics: Company Search
    elif "Last Funding Type" in df:
        print(' * detected a Company Search CSV')
        df.rename(columns={
            "Organization Name": COL_NAME,
            "Last Funding Type": COL_SERIES,
            "Last Funding Date": COL_FUND_DATE,
            # Money
            "Industries": COL_INDUSTRIES,
            "Description": COL_DESCRIPTION,
        }, inplace=True)
        df[COL_MONEY] = 0

    # type heuristics: Company List (BARE MINIMUM)
    elif "Organization Name" in df and "Industries" in df and "Description" in df:
        print(' * detected a BARE MINIMUM Company List CSV')
        df.rename(columns={
            "Organization Name": COL_NAME,
            "Industries": COL_INDUSTRIES,
            "Description": COL_DESCRIPTION,
        }, inplace=True)
        df[COL_SERIES] = 'Unknown'
        df[COL_FUND_DATE] = 'Unknown'
        df[COL_MONEY] = 0

    # type heuristics: ?
    else:
        raise Exception('Wrong CSV file type')

    if COL_TITLE not in df:
        df[COL_TITLE] = df.apply(lambda row: row[COL_NAME] + ((' (' + str(round(row[COL_MONEY])) + ' M)') if np.isfinite(row[COL_MONEY]) else ''), axis=1)
    if COL_FUND_YEAR not in df:
        df[COL_FUND_YEAR] = df.apply(lambda row: row[COL_FUND_DATE][:4] if row[COL_FUND_DATE] != 'Unknown' and row[COL_FUND_DATE] == row[COL_FUND_DATE] else '', axis=1)
    if COL_LABEL not in df:
        df[COL_LABEL] = '_'

    # add optional columns, if present in the dataset
    headers = _TSV_HEADERS.copy()
    for col in _TSV_OPTIONALS:
        if col in df and col not in headers:
            headers.append(col)
    return df[headers], headers
