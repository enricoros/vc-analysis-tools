"""
Enrico 2021 - NLP sentence-embeddings service, exposed as REST endpoint
"""
import csv
import json
import traceback
from io import BytesIO, StringIO

import fire as fire
import numpy
import pandas as pd
from flask import Flask, render_template, request, send_file
from flask_cors import cross_origin

from utils_crunchy import normalize_crunchbase_df, COL_INDUSTRIES, COL_DESCRIPTION
from utils_embeddings import text_to_embeds_use, text_to_embeds_use_fast, text_to_embeds_mpnet

# configuration
default_http_address = '127.0.0.1'
default_http_port = 8000
default_api_prefix = '/embeds'

page_upload_html_resp = '/upload_csv'
page_download = '/download'
page_last_results = '/last_results'

# In-Mem-Downloads - FIXME: have some purge strategy, this is just a mega-leaker
in_mem_downloads = {}
in_mem_results = []


# Flash main app
def run_app(http_host=default_http_address, http_port=default_http_port, api_prefix=default_api_prefix):
    # configure Flask for serving
    print(f'\n# Starting HTTP endpoint on {http_host}: {http_port}, api prefix: {api_prefix}')
    app = Flask(__name__)
    app.logger.setLevel(20)
    print()

    # warm up the predictors
    text_to_embeds_use(['House', 'Home', 'Cat'])
    text_to_embeds_mpnet(['House', 'Home', 'Cat'])

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
        df_cb, df_headers = normalize_crunchbase_df(df_cb)

        # select the nlp column set (dynamic)
        nlp_cols = [COL_INDUSTRIES]
        if 'col' in request.form:
            col_form = int(request.form['col'])
            if col_form == 0:
                nlp_cols = [COL_INDUSTRIES]
            elif col_form == 1:
                nlp_cols = [COL_DESCRIPTION]
            elif col_form == 2:
                nlp_cols = [COL_INDUSTRIES, COL_DESCRIPTION]
            elif col_form == 3:
                nlp_cols = ['Ind + Desc']
                df_cb['Ind + Desc'] = 'Industries: ' + df_cb[COL_INDUSTRIES] + '. Description: ' + df_cb[COL_DESCRIPTION]
            else:
                print(f'EE: embedding columns requested ({col_form}) is not supported. Using default.')
        if len(nlp_cols) < 1:
            raise Exception(f'Unspecified NLP columns')
        for col in nlp_cols:
            if col not in df_cb:
                raise Exception(f'Cannot find the "{col}" field in the data set.')
            df_cb.dropna(subset=[col], inplace=True)

        # select the model (dynamic)
        model_fun = text_to_embeds_use
        if 'model' in request.form:
            model_form = int(request.form['model'])
            if model_form == 0:
                model_fun = text_to_embeds_use
            elif model_form == 1:
                model_fun = text_to_embeds_use_fast
            elif model_form == 2:
                model_fun = text_to_embeds_mpnet
            else:
                print(f'EE: model requested ({model_form}) is not supported. Fallback to using: {model_fun}')

        # perform the NLP analysis, concatenating all the embeds in the provided columns
        model_name = None
        companies_embeds = None
        for col in nlp_cols:
            nlp_strings = list(df_cb[col])
            model_name, col_embeds, _ = model_fun(nlp_strings)
            if companies_embeds is None:
                companies_embeds = col_embeds
            else:
                companies_embeds = numpy.concatenate((companies_embeds, col_embeds), axis=1)

        # export as TSV
        nlp_fields = "-".join(nlp_cols)
        investor_name = csv_name.replace('data/', '').split('-')[0].capitalize()
        file_base_name = f'embeds-{nlp_fields}-{model_name}-{investor_name}'
        analysis_title_name = f'{investor_name}-{nlp_fields} ({model_name})'

        # metadata
        companies_meta = df_cb[df_headers].to_numpy()
        return file_base_name, analysis_title_name, companies_embeds, companies_meta, df_headers, nlp_fields

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
        return render_template('embeds_simple_frontend.html', api_prefix=api_prefix)

    @app.route(api_prefix + page_upload_html_resp, methods=['POST'])
    @cross_origin()
    def analyze_csv():
        global in_mem_downloads, in_mem_results
        try:
            file_base_name, analysis_title_name, companies_embeds, companies_meta, meta_headers, nlp_field = process_uploaded_file()

            embeds_uid = f'{file_base_name}.tsv'
            embeds_tsv = array_to_tsv_string(companies_embeds, embeds_uid)
            meta_uid = f'{file_base_name}-meta.tsv'
            meta_tsv = array_to_tsv_string(companies_meta, meta_uid, meta_headers)
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

            # API response JSON
            result = {'embeds': {'name': embeds_uid, 'length': len(embeds_tsv), 'shape': companies_embeds.shape, 'nlp_field': nlp_field},
                      'meta': {'name': meta_uid, 'length': len(meta_tsv), 'shape': companies_meta.shape, 'fields': meta_headers},
                      'config': {'name': config_uid}}

            # NOTE: this replaces the full contents, so former generations will not be accessible
            # HACK: shall cache-purge, but we're keeping just the last item, instead
            # hack_in_mem_downloads = {
            #     embeds_uid: embeds_tsv,
            #     meta_uid: meta_tsv,
            #     config_uid: json.dumps(config_obj)
            # }
            # NOTE: keep in-memory - FIXME: needs some eviction policy/algo
            in_mem_downloads[embeds_uid] = embeds_tsv
            in_mem_downloads[meta_uid] = meta_tsv
            in_mem_downloads[config_uid] = json.dumps(config_obj)
            in_mem_results.append(result)
            return result, 200

        except Exception as e:
            print("EXCEPTION on " + page_upload_html_resp)
            traceback.print_exc()
            return {"backend_exception": repr(e)}, 500

    @app.route(api_prefix + page_last_results, methods=['GET'])
    @cross_origin()
    def get_last_results():
        try:
            global in_mem_results
            return {'results': in_mem_results}, 200

        except Exception as e:
            print("EXCEPTION on " + page_last_results)
            traceback.print_exc()
            return {"backend_exception": repr(e)}, 500

    @app.route(api_prefix + page_download + '/<name>', methods=['GET'])
    @cross_origin()
    def download_from_cache(name):
        try:
            if name not in in_mem_downloads:
                raise Exception('File Unknown')
            print(f'...serving {name}')
            contents = in_mem_downloads[name]

            # send contents as file (requires temp stream)
            buffer = BytesIO()
            buffer.write(contents.encode())
            buffer.seek(0)
            return send_file(buffer, as_attachment=True, download_name=name, mimetype='text/csv')

        except Exception as e:
            print("EXCEPTION on " + page_download)
            traceback.print_exc()
            return {"backend_exception": repr(e)}, 500

    # run the event loop here
    app.run(host=http_host, port=http_port, threaded=False)


if __name__ == '__main__':
    fire.Fire(run_app)
