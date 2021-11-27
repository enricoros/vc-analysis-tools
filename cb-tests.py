import fire as fire
import numpy as np
import pandas as pd
from metadata_parser import MetadataParser

from utils_crunchy import normalize_crunchbase_df, COL_FUND_YEAR


def test1(cb_bulk_folder: str):
    df_orgs = pd.read_csv(f'{cb_bulk_folder}/organizations.csv')
    df_descs = pd.read_csv(f'{cb_bulk_folder}/organization_descriptions.csv')

    orgs_uuid = df_orgs['uuid']
    descs_uuid = df_descs['uuid']

    # a = np.intersect1d(orgs_uuid.to_list(), descs_uuid.to_list())

    # sort both uuid series
    a = np.sort(orgs_uuid.to_list())

    b = np.sort(descs_uuid.to_list())

    pass


def main():
    try:
        url = 'https://hu.ma.ne'
        parser: MetadataParser = MetadataParser(url, search_head_only=False, support_malformed=True, ssl_verify=True)
        result = parser.parsed_result
        print(result.get_metadatas('description'))
    except Exception as e:
        print(e)
    pass


if __name__ == '__main__':
    fire.Fire(main)
