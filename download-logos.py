"""
Enrico 2021
"""
import os
import re
import shutil
import time
from pathlib import Path

import cloudinary
import fire as fire
import magic
import pandas as pd
import requests
from bs4 import BeautifulSoup
from numpy import random

from analyze import normalize_crunchbase_df, COL_NAME

COL_CB_PAGE = 'Organization Name URL'

REQ_HEADERS = {
    'accept-language': 'en-US,en;q=0.9',
    'dnt': '1',
    'upgrade-insecure-requests': '1',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
}

EXT_PNG = '.png'
EXT_JPG = '.jpeg'


def download_html(url):
    res = requests.get(url, headers=REQ_HEADERS)
    if res.status_code == 200:
        return res.content.decode()
    raise Exception(f'Error fetching HTML: {url}', res.status_code, res.text)


def download_image(url, file_name):
    res = requests.get(url, stream=True)
    if res.status_code == 200:
        res.raw.decode_content = True
        with open(file_name, 'wb') as f:
            shutil.copyfileobj(res.raw, f)
        return
    raise Exception(f'Error fetching Image to file: {url}', res.status_code, res.text)


def parse_og_image_in_html(html_text):
    soup = BeautifulSoup(html_text, 'html.parser')
    for meta_tag in soup.findAll('meta'):
        meta_attrs = meta_tag.attrs
        if len(meta_attrs) != 2:
            raise Exception(f'<meta> has {len(meta_attrs)} attributes - expected 2', meta_attrs)
        if 'name' in meta_attrs:
            # ignore the 3 name attributes: viewport, google-site-verification and description (duplicate)
            continue
        elif 'property' in meta_attrs:
            # only use 'og:image' and ignore everything else
            meta_property = meta_attrs['property']
            if meta_property == 'og:image':
                return meta_attrs['content']
        else:
            raise Exception(f'<meta> of unknown kind (exp: name or property)', meta_attrs)
    raise Exception(f'<meta> og:image not found')


def run_app(csv_file, out_folder=''):
    # create destination folder, if not present
    Path(out_folder).mkdir(parents=True, exist_ok=True)

    # read the file, assuming is CB
    df_cb = pd.read_csv(csv_file)

    # validate the crunchbase-ness
    try:
        normalize_crunchbase_df(df_cb)
        if COL_CB_PAGE not in df_cb:
            raise Exception(f'missing column {COL_CB_PAGE}')
    except Exception as e:
        print("ERROR: may not be a valid Crunchbase file", e)
        exit(1)

    # ready cloudinary api for crunchbase download
    cloudinary.config(cloud_name="crunchbase-production")

    # process each line
    for index, org_row in df_cb.iterrows():
        org_name = org_row[COL_NAME]
        # org_description = org_row[COL_DESCRIPTION]
        org_url = org_row[COL_CB_PAGE]

        # logo file name
        org_logo_file_name = re.sub("[^0-9a-zA-Z]+", "-", org_name.lower())
        org_logo_file_path = os.path.join(out_folder, org_logo_file_name)
        if os.path.isfile(org_logo_file_path + EXT_PNG) or os.path.isfile(org_logo_file_path + EXT_JPG):
            print(f' - skipping: {org_logo_file_path}')
            continue

        # load the CB organization URL to find more info from the META of the HTML
        org_page_html = download_html(org_url)
        cloudinary_image_url = parse_og_image_in_html(org_page_html)  # example: 'https://res.cloudinary.com/crunchbase-production/image/upload/c_lpad,h_256,w_256,f_auto,q_auto:eco,dpr_1/pwzuoya5pdebfii6bfbg'
        cloudinary_image_suffix = cloudinary_image_url.split('/')[-1]  # example: pwzuoya5pdebfii6bfbg

        # download the image, original, without transformations (highest possible quality)
        cloudinary_image_orig_url = cloudinary.CloudinaryImage(cloudinary_image_suffix).build_url(transformations=[])  # example: 'http://res.cloudinary.com/crunchbase-production/image/upload/pwzuoya5pdebfii6bfbg'
        download_image(cloudinary_image_orig_url, org_logo_file_path)
        org_logo_mime = magic.from_file(org_logo_file_path, mime=True)
        if org_logo_mime != 'image/jpeg':
            os.rename(org_logo_file_path, org_logo_file_path + EXT_JPG)
        elif org_logo_mime != 'image/png':
            os.rename(org_logo_file_path, org_logo_file_path + EXT_PNG)
        else:
            raise Exception(f'unsupported image format: {org_logo_mime}, for {org_logo_file_path}')

        # done
        print(f' + [{index + 1}/{df_cb.shape[0]}] downloaded: {org_logo_file_path}, of type', org_logo_mime)

        # do not overload the server with requests
        time.sleep(random.uniform(5, 10) + random.uniform(1, 10))


if __name__ == '__main__':
    fire.Fire(run_app)
