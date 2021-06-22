"""
Enrico 2021 - NLP sentence-embeddings service, exposed as REST endpoint
"""
import csv
import traceback
from io import BytesIO, StringIO

import fire as fire
import pandas as pd
from flask import Flask, render_template, request, send_file

from analyze import normalize_crunchbase_df, text_to_embeds_use, TSV_HEADERS, COL_INDUSTRIES

# configuration
default_http_address = '127.0.0.1'
default_http_port = 1900
default_api_prefix = '/embeds'

page_upload_html_resp = '/upload_csv'
page_download = '/download_tsv/<name>'

# In-Mem-Downloads - FIXME: have some purge strategy, this is just a mega-leaker
hack_in_mem_downloads = {}


# Flash main app
def run_app(http_host=default_http_address, http_port=default_http_port, api_prefix=default_api_prefix):
    # configure Flask for serving
    print(f'\n# Starting HTTP endpoint on {http_host}: {http_port}, api prefix: {api_prefix}')
    app = Flask(__name__)
    app.logger.setLevel(20)
    print()

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
        if COL_INDUSTRIES not in df_cb:
            raise Exception('Cannot find the "Industries" field')
        col_nlp = COL_INDUSTRIES
        df_cb.dropna(subset=[col_nlp], inplace=True)

        # perform the NLP analysis (first time it will load the model)
        nlp_strings = list(df_cb[col_nlp])
        model_name, companies_embeds, companies_corr = text_to_embeds_use(nlp_strings)

        # export as TSV
        investor_name = csv_name.replace('data/', '').split('-')[0].capitalize()
        file_base_name = f'embeds-{col_nlp}-{model_name}-{investor_name}'

        # metadata
        companies_meta = df_cb[TSV_HEADERS].to_numpy()
        return file_base_name, companies_embeds, companies_meta

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
    def analyze_csv():
        global hack_in_mem_downloads
        try:
            file_base_name, companies_embeds, companies_meta = process_uploaded_file()

            embeds_uid = f'{file_base_name}.tsv'
            embeds_tsv = array_to_tsv_string(companies_embeds, embeds_uid)
            meta_uid = f'{file_base_name}-meta.tsv'
            meta_tsv = array_to_tsv_string(companies_meta, meta_uid, TSV_HEADERS)

            # NOTE: this replaces the full contents, so former generations will not be accessible
            # HACK: shall cache-purge, but we're keeping just the last item, instead
            hack_in_mem_downloads = {embeds_uid: embeds_tsv, meta_uid: meta_tsv}

            return {'embeds': {
                'name': embeds_uid, 'length': len(embeds_tsv), 'shape': companies_embeds.shape,
            }, 'meta': {
                'name': meta_uid, 'length': len(meta_tsv), 'shape': companies_meta.shape,
            }}, 200

            # return f"""<html>
            # <body>
            #   <div>Outputs - "save as..." the following:
            #   <ul>
            #     <li><a href="{api_prefix}/download_tsv/{embeds_uid}">{embeds_uid}</a> ({len(embeds_tsv)} bytes)</li>
            #     <li><a href="{api_prefix}/download_tsv/{meta_uid}">{meta_uid}</a> ({len(meta_tsv)} bytes)</li>
            #   </ul>
            # </body>
            # </html>""", 200

        except Exception as e:
            print("EXCEPTION on analyze_csv")
            traceback.print_exc()
            return {"backend_exception": repr(e)}, 500

    @app.route(api_prefix + page_download, methods=['GET'])
    def download_tsv(name):
        try:
            if name not in hack_in_mem_downloads:
                raise Exception('File Unknown')
            print(f'...serving {name}')
            contents = hack_in_mem_downloads[name]

            # send contents as file (requires temp stream)
            buffer = BytesIO()
            buffer.write(contents.encode())
            buffer.seek(0)
            return send_file(buffer, as_attachment=True, attachment_filename=name, mimetype='text/csv')

        except Exception as e:
            print("EXCEPTION on download_tsv")
            traceback.print_exc()
            return {"backend_exception": repr(e)}, 500

    # run the event loop here
    app.run(host=http_host, port=http_port, threaded=False)


if __name__ == '__main__':
    fire.Fire(run_app)
