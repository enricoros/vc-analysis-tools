"""
Enrico 2021 - NLP sentence-embeddings service, exposed as REST endpoint
"""
import csv
import json
import traceback
from io import BytesIO, StringIO

import fire as fire
import pandas as pd
from flask import Flask, render_template, request, send_file
from flask_cors import cross_origin

from analyze import normalize_crunchbase_df, text_to_embeds_use, TSV_HEADERS, COL_INDUSTRIES, COL_DESCRIPTION

# configuration
default_http_address = '127.0.0.1'
default_http_port = 1900
default_api_prefix = '/embeds'

page_upload_html_resp = '/upload_csv'
page_download = '/download'

# In-Mem-Downloads - FIXME: have some purge strategy, this is just a mega-leaker
hack_in_mem_downloads = {}


# Flash main app
def run_app(http_host=default_http_address, http_port=default_http_port, api_prefix=default_api_prefix):
    # configure Flask for serving
    print(f'\n# Starting HTTP endpoint on {http_host}: {http_port}, api prefix: {api_prefix}')
    app = Flask(__name__)
    app.logger.setLevel(20)
    print()

    # warm up the predictor
    text_to_embeds_use(['House', 'Home', 'Cat'])

    # load the file received as attachment, produce the embeds, prepare the 2 data arrays
    def process_uploaded_file():
        # load the CSV file into memory
        if 'file' not in request.files:
            raise Exception('Missing Attachment')
        f = request.files['file']
        csv_name = f.filename
        csv_contents = f.stream.read()
        if len(csv_contents) < 1:
            raise Exception('Short CSV file')

        # Normalized DataFrame from the CSV
        df_cb = pd.read_csv(BytesIO(csv_contents))
        df_cb = normalize_crunchbase_df(df_cb)
        # find the NLP column
        col_nlp = COL_INDUSTRIES
        if 'col' in request.form:
            col_form = request.form['col']
            if col_form == '0':
                col_nlp = COL_INDUSTRIES
            elif col_form == '1':
                col_nlp = COL_DESCRIPTION
            else:
                print(f'EE: embedding columns requested ({col_form}) is not supported. Fallback to using: {col_nlp}')
        if col_nlp not in df_cb:
            raise Exception(f'Cannot find the "{col_nlp}" field')
        df_cb.dropna(subset=[col_nlp], inplace=True)

        # perform the NLP analysis (first time it will load the model)
        nlp_strings = list(df_cb[col_nlp])
        model_name, companies_embeds, companies_corr = text_to_embeds_use(nlp_strings)

        # export as TSV
        investor_name = csv_name.replace('data/', '').split('-')[0].capitalize()
        file_base_name = f'embeds-{col_nlp}-{model_name}-{investor_name}'
        analysis_title_name = f'{investor_name}-{col_nlp} ({model_name})'

        # metadata
        companies_meta = df_cb[TSV_HEADERS].to_numpy()
        return file_base_name, analysis_title_name, companies_embeds, companies_meta, col_nlp

    # numpy array to tsv (csv) string, with optional headers
    def array_to_tsv_string(array, tsv_name, headers=None):
        print(f' - exported: {tsv_name}')
        string = StringIO()
        writer = csv.writer(string, delimiter='\t', lineterminator='\n')
        if headers is not None:
            writer.writerow(headers)
        for row in array:
            writer.writerow(row.tolist())
        return string.getvalue()

    @app.route(api_prefix + '/index.html', methods=['GET'])
    def upload_file():
        return render_template('analysis_service_upload.html', api_prefix=api_prefix)

    @app.route(api_prefix + page_upload_html_resp, methods=['POST'])
    @cross_origin()
    def analyze_csv():
        global hack_in_mem_downloads
        try:
            file_base_name, analysis_title_name, companies_embeds, companies_meta, nlp_field = process_uploaded_file()

            embeds_uid = f'{file_base_name}.tsv'
            embeds_tsv = array_to_tsv_string(companies_embeds, embeds_uid)
            meta_uid = f'{file_base_name}-meta.tsv'
            meta_tsv = array_to_tsv_string(companies_meta, meta_uid, TSV_HEADERS)
            config_uid = f'{file_base_name}-config.json'
            config_obj = {
                "embeddings": [
                    {
                        "tensorName": analysis_title_name + ' Analysis',
                        "tensorShape": [
                            companies_embeds.shape[0],  # 12 companies
                            companies_embeds.shape[1],  # 512 embeds
                        ],
                        "tensorPath": f'https://www.enrico.ai{api_prefix}{page_download}/{embeds_uid}',
                        "metadataPath": f'https://www.enrico.ai{api_prefix}{page_download}/{meta_uid}',
                    }
                ]
            }

            # NOTE: this replaces the full contents, so former generations will not be accessible
            # HACK: shall cache-purge, but we're keeping just the last item, instead
            hack_in_mem_downloads = {
                embeds_uid: embeds_tsv,
                meta_uid: meta_tsv,
                config_uid: json.dumps(config_obj)
            }

            return {'embeds': {'name': embeds_uid, 'length': len(embeds_tsv), 'shape': companies_embeds.shape, 'nlp_field': nlp_field},
                    'meta': {'name': meta_uid, 'length': len(meta_tsv), 'shape': companies_meta.shape, 'fields': TSV_HEADERS},
                    'config': {'name': config_uid}}, 200

            # return f"""<html>
            # <body>
            #   <div>Outputs - "save as..." the following:
            #   <ul>
            #     <li><a href="{api_prefix}/download/{embeds_uid}">{embeds_uid}</a> ({len(embeds_tsv)} bytes)</li>
            #     <li><a href="{api_prefix}/download/{meta_uid}">{meta_uid}</a> ({len(meta_tsv)} bytes)</li>
            #   </ul>
            # </body>
            # </html>""", 200

        except Exception as e:
            print("EXCEPTION on analyze_csv")
            traceback.print_exc()
            return {"backend_exception": repr(e)}, 500

    @app.route(api_prefix + page_download + '/<name>', methods=['GET'])
    @cross_origin()
    def download_from_cache(name):
        try:
            if name not in hack_in_mem_downloads:
                raise Exception('File Unknown')
            print(f'...serving {name}')
            contents = hack_in_mem_downloads[name]

            # send contents as file (requires temp stream)
            buffer = BytesIO()
            buffer.write(contents.encode())
            buffer.seek(0)
            return send_file(buffer, as_attachment=True, download_name=name, mimetype='text/csv')

        except Exception as e:
            print("EXCEPTION on download")
            traceback.print_exc()
            return {"backend_exception": repr(e)}, 500

    # run the event loop here
    app.run(host=http_host, port=http_port, threaded=False)


if __name__ == '__main__':
    fire.Fire(run_app)
