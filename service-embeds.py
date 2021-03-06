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
from utils_embeddings import text_to_embeds_use_large, text_to_embeds_use_fast, text_to_embeds_mpnet

# configuration
default_http_address = '127.0.0.1'
default_http_port = 8000
default_api_prefix = '/embeds'

page_home = '/index.html'
page_list_models = '/list_models'
page_analyze_csv = '/analyze_csv'
page_analyze_json = '/analyze_json'
page_download = '/download'
page_recent = '/last_results'

# Supported CPU models for embedding generation
MODELS = [
    {'num': 0, 'name': 'use-large-5', 'fun': text_to_embeds_use_large},
    {'num': 1, 'name': 'use-fast-4', 'fun': text_to_embeds_use_fast},
    {'num': 2, 'name': 'all-mpnet-base-v2', 'fun': text_to_embeds_mpnet},
]

# In-Mem-Downloads - FIXME: have some purge strategy, this is just a mega-leaker
in_mem_downloads = {}
in_mem_results = []


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


# acquires the correct engine
def model_num_to_engine(model_num_or_name):
    model = next((m for m in MODELS if (m['num'] == model_num_or_name or m['name'] == model_num_or_name)), None)
    if model is None:
        print(f'EE: model requested ({model_num_or_name}) is not supported. Fallback to using 0')
        return text_to_embeds_use_large
    return model['fun']


# load the file received as attachment, produce the embeds, prepare the 2 data arrays
def process_uploaded_file(csv_contents, investor_name, col_num, model_num):
    # Normalized DataFrame from the CSV
    df_cb = pd.read_csv(BytesIO(csv_contents))
    df_cb, df_headers = normalize_crunchbase_df(df_cb)

    # select the nlp column set (dynamic)
    nlp_cols = [COL_INDUSTRIES]
    if col_num == 0:
        nlp_cols = [COL_INDUSTRIES]
    elif col_num == 1:
        nlp_cols = [COL_DESCRIPTION]
    elif col_num == 2:
        nlp_cols = [COL_INDUSTRIES, COL_DESCRIPTION]
    elif col_num == 3:
        nlp_cols = ['Ind + Desc']
        df_cb['Ind + Desc'] = 'Industries: ' + df_cb[COL_INDUSTRIES] + '. Description: ' + df_cb[COL_DESCRIPTION]
    else:
        print(f'EE: embedding columns requested ({col_num}) is not supported. Using default.')
    if len(nlp_cols) < 1:
        raise Exception(f'Unspecified NLP columns')
    for col in nlp_cols:
        if col not in df_cb:
            raise Exception(f'Cannot find the "{col}" field in the data set.')
        df_cb.dropna(subset=[col], inplace=True)

    # select the model (dynamic)
    model_fun = model_num_to_engine(model_num)

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
    file_base_name = f'embeds-{nlp_fields}-{model_name}-{investor_name}'
    analysis_title_name = f'{investor_name}-{nlp_fields} ({model_name})'

    # metadata
    companies_meta = df_cb[df_headers].to_numpy()
    return file_base_name, analysis_title_name, companies_embeds, companies_meta, df_headers, nlp_fields


# Flash main app
def run_app(http_host=default_http_address, http_port=default_http_port, api_prefix=default_api_prefix,
            exit_after_warm=False):
    # warm up the predictors
    text_to_embeds_use_large(['House', 'Home', 'Cat'])
    text_to_embeds_mpnet(['House', 'Home', 'Cat'])
    if exit_after_warm is not False:
        print('Warmed up successfully. Exiting.')
        return

    # configure Flask for serving
    print(f'\n# Starting HTTP endpoint on {http_host}: {http_port}, api prefix: {api_prefix}')
    app = Flask(__name__)
    app.logger.setLevel(20)
    print()

    @app.route(api_prefix + page_home, methods=['GET'])
    def render_home():
        # noinspection PyUnresolvedReferences
        return render_template('embeds_simple_frontend.html', api_prefix=api_prefix)

    @app.route(api_prefix + page_list_models, methods=['GET'])
    @cross_origin()
    def list_models():
        return {"engines": [m['name'] for m in MODELS]}, 200

    @app.route(api_prefix + page_analyze_json, methods=['POST'])
    @cross_origin()
    def analyze_json():
        try:
            # get and parse the input { model: 1, strings: [...] }
            json_data = request.json
            model_num = json_data['model']  # can be the number or the name (from /list_models)
            nlp_strings = json_data['input']  # list of strings

            # compute embeddings locally
            model_fun = model_num_to_engine(model_num)
            model_name, embeddings, _ = model_fun(nlp_strings)

            # API response JSON
            # NOTE: we are limiting the precision of the embeddings to 6 decimals (from 19)
            result = {
                'embeds': numpy.round(embeddings.astype(float), 9).tolist(),
                'model_name': model_name,
                'dimensions': embeddings.shape[1],
                'shape': list(embeddings.shape),
            }
            return result

        except Exception as e:
            print("EXCEPTION on " + page_analyze_json)
            traceback.print_exc()
            return {"backend_exception": repr(e)}, 500

    @app.route(api_prefix + page_analyze_csv, methods=['POST'])
    @cross_origin()
    def analyze_csv():
        global in_mem_downloads, in_mem_results
        try:
            # get the attached file
            if 'file' not in request.files:
                raise Exception('Missing Attachment')
            file = request.files['file']
            csv_name = file.filename
            investor_name = csv_name.replace('data/', '').split('-')[0].capitalize()
            csv_contents = file.stream.read()
            if len(csv_contents) < 1:
                raise Exception('Short CSV file')

            # which col to operate on
            col_num = 0
            if 'col' in request.form:
                col_num = int(request.form['col'])

            # which embeddings model to use
            model_num = 0
            if 'model' in request.form:
                model_num = int(request.form['model'])

            file_base_name, analysis_title_name, companies_embeds, companies_meta, meta_headers, nlp_field = process_uploaded_file(
                csv_contents, investor_name, col_num, model_num)

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
            result = {'embeds': {'name': embeds_uid, 'length': len(embeds_tsv), 'shape': companies_embeds.shape,
                                 'nlp_field': nlp_field},
                      'meta': {'name': meta_uid, 'length': len(meta_tsv), 'shape': companies_meta.shape,
                               'fields': meta_headers},
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
            print("EXCEPTION on " + page_analyze_csv)
            traceback.print_exc()
            return {"backend_exception": repr(e)}, 500

    @app.route(api_prefix + page_download + '/<name>', methods=['GET'])
    @cross_origin()
    def download_from_memory(name):
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

    @app.route(api_prefix + page_recent, methods=['GET'])
    @cross_origin()
    def get_recent_results():
        try:
            global in_mem_results
            return {'results': in_mem_results}, 200

        except Exception as e:
            print("EXCEPTION on " + page_recent)
            traceback.print_exc()
            return {"backend_exception": repr(e)}, 500

    # run the event loop here
    app.run(host=http_host, port=http_port, threaded=False)


if __name__ == '__main__':
    fire.Fire(run_app)
