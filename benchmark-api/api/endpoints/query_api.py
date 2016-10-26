import json

from flask.ext.api import status
import flask as fk

from api import app, API_URL, crossdomain, api_response
from benchdb.common.models import RowModel
from benchdb.common.models import ColModel
from benchdb.common.models import DescModel

import mimetypes
import json
import traceback
import datetime
import random
import string
import os
import _thread
from io import StringIO
import pandas as pd


@app.route(API_URL + '/push/csv', methods=['GET','POST','PUT','UPDATE','DELETE'])
@crossdomain(origin='*')
def home_reference_evaluate_data():
    if fk.request.method == 'POST':
        if fk.request.files:
            file_obj = fk.request.files['file']
            file_name = file_obj.filename
            dataObject = StringIO()

            head = RowModel.objects(index='0').first()

            with open('/tmp/{0}'.format(file_name), 'w') as tmp_csv:
                content = file_obj.read()
                tmp_csv.write(content)

            print("New file witten to tmp...")

            data = pd.read_csv('/tmp/{0}'.format(file_name))

            if head == None:
                head = RowModel(index='0')
                head.value = [ col for col in data.columns]
                head.save()

            print("Head section loaded...")
            header = ','.join(head.value)
            print('%s\n'%header)

            previous_index = len(RowModel.objects())

            for index, row in data.iterrows():
                values = []
                for c in head.value:
                    values.append(row[c])
                rw = RowModel(index=str(previous_index+index), value=values)
                rw.save()

            print("New data appended to old data...")

           
            dataObject.write('%s\n'%header)

            print("Head written to dataframe...")

            for row in RowModel.objects():
                if int(row.index) > 0:
                    oneline = ','.join([str(v) for v in row.value])
                    dataObject.write('%s\n'%oneline)

            print("New merged content written to dataframe...")

            dataObject.seek(0)
            data_merged = pd.read_csv(dataObject)

            for desc in DescModel.objects():
                desc.delete()

            for col in ColModel.objects():
                col.delete()

            print("Previous decription and columns deleted...")

            for c in head.value:
                _desc = data_merged[c].describe()
                desc = DescModel(column=c)
                col = ColModel(column=c)

                col.values = data_merged[c].tolist()
                col.save()
                description = {}
                description['count'] = _desc['count']
                description['dtype'] = str(_desc.dtype)
                if description['dtype'] == 'float64':
                    description['mean'] = _desc['mean']
                    description['std'] = _desc['std']
                    description['min'] = _desc['min']
                    description['max'] = _desc['max']
                    description['25%'] = _desc['min']
                    description['50%'] = _desc['min']
                    description['75%'] = _desc['min']
                    description['histogram'] = data[c]
                elif description['dtype'] == 'object':
                    description['unique'] = _desc['unique']
                    description['top'] = _desc['top']
                    description['freq'] = _desc['freq']
                    description['options'] = data_merged[c].unique().tolist()
                elif description['dtype'] == 'datetime64':
                    description['unique'] = _desc['unique']
                    description['options'] = data_merged[c].unique().tolist()
                    description['first'] = _desc['first']
                    description['last'] = _desc['last']
                desc.df = description
                desc.save()
            print("New description and columns added...")
            return api_response(200, 'Push succeed', 'Your file was pushed.')
        else:
            return api_response(204, 'Nothing created', 'You must a set file.')

    return """
    <!doctype html>
    <html>
        <head>
          <!-- css  -->
          <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
          <link href="http://0.0.0.0:4000/css/materialize.min.css" type="text/css" rel="stylesheet" media="screen,projection"/>
          <link href="http://0.0.0.0:4000/css/style.css" type="text/css" rel="stylesheet" media="screen,projection"/>

          <!--Let browser know website is optimized for mobile-->
          <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1.0, user-scalable=no"/>
          <title>Benchmark Platform</title>
        </head>
        <body>
            <nav class="white" role="navigation">
                <div class="nav-wrapper container">
                  <a id="logo-container" href="http://0.0.0.0:8000" class="teal-text text-lighten-2">Reference</a>
                </div>
            </nav>
            <div class="valign center-align">
                <h1>Upload dataset</h1>
                <form action="" method=post enctype=multipart/form-data>
                    <input type=file name=file>
                    <input type=submit value=Upload>
                </form>
            </div>
            <!--Import jQuery before materialize.js-->
            <script type="text/javascript" src="https://code.jquery.com/jquery-2.1.1.min.js"></script>
            <script type="text/javascript" src="http://0.0.0.0:4000/js/materialize.min.js"></script>
        </body>
    </html>
    """

@app.route(API_URL + '/desc/all', methods=['GET','POST','PUT','UPDATE','DELETE'])
@crossdomain(origin='*')
def query_desc_all():
    if fk.request.method == 'GET':
        _descriptions = DescModel.objects()
        descriptions = []
        for desc in _descriptions:
            descriptions.append(desc.info())
        return api_response(200, 'Columns descriptions', descriptions)
    else:
        return api_response(405, 'Method not allowed', 'This endpoint supports only a GET method.')

@app.route(API_URL + '/desc/single/<column>', methods=['GET','POST','PUT','UPDATE','DELETE'])
@crossdomain(origin='*')
def query_desc_single(column):
    if fk.request.method == 'GET':
        description = DescModel.objects(column=column).first()
        if description:
            return api_response(200, 'Column [%s] description'%column, description.df)
        else:
            return api_response(204, 'Nothing found', "No column with that name.")
    else:
        return api_response(405, 'Method not allowed', 'This endpoint supports only a GET method.')


@app.route(API_URL + '/clear', methods=['GET','POST','PUT','UPDATE','DELETE'])
@crossdomain(origin='*')
def clear():
    if fk.request.method == 'GET':
        descriptions = DescModel.objects()
        for desc in descriptions:
            desc.delete()
        cols = ColModel.objects()
        for col in cols:
            col.delete()
        rws = RowModel.objects()
        for rw in rws:
            rw.delete()
        return api_response(204, 'Clear done', "Everything is wiped out.")
    else:
        return api_response(405, 'Method not allowed', 'This endpoint supports only a GET method.')

@app.route(API_URL + '/col/all', methods=['GET','POST','PUT','UPDATE','DELETE'])
@crossdomain(origin='*')
def query_col_all():
    if fk.request.method == 'GET':
        _cols = ColModel.objects()
        cols = []
        for col in _cols:
            cols.append(col.info())
        return api_response(200, 'Columns values', cols)
    else:
        return api_response(405, 'Method not allowed', 'This endpoint supports only a GET method.')

@app.route(API_URL + '/col/dict/all', methods=['GET','POST','PUT','UPDATE','DELETE'])
@crossdomain(origin='*')
def query_col_dict_all():
    if fk.request.method == 'GET':
        _cols = ColModel.objects()
        cols = {}
        for col in _cols:
            info = col.info()
            cols[col.column] = col.values
        return api_response(200, 'Columns as dicts', cols)
    else:
        return api_response(405, 'Method not allowed', 'This endpoint supports only a GET method.')

@app.route(API_URL + '/col/dict/bare', methods=['GET','POST','PUT','UPDATE','DELETE'])
@crossdomain(origin='*')
def query_col_dict_bare():
    if fk.request.method == 'GET':
        _cols = ColModel.objects()
        cols = {}
        for col in _cols:
            info = col.info()
            cols[col.column] = []
        return api_response(200, 'Columns as bare dicts', cols)
    else:
        return api_response(405, 'Method not allowed', 'This endpoint supports only a GET method.')

@app.route(API_URL + '/col/single/<column>', methods=['GET','POST','PUT','UPDATE','DELETE'])
@crossdomain(origin='*')
def query_single_single(column):
    if fk.request.method == 'GET':
        col = ColModel.objects(column=column).first()
        if col:
            return api_response(200, 'Column [%s] content'%column, col.info())
        else:
            return api_response(204, 'Nothing found', "No column with that name.")
    else:
        return api_response(405, 'Method not allowed', 'This endpoint supports only a GET method.')

@app.route(API_URL + '/row/all', methods=['GET','POST','PUT','UPDATE','DELETE'])
@crossdomain(origin='*')
def query_row_all():
    if fk.request.method == 'GET':
        _rws = RowModel.objects()
        rws = []
        for rw in _rws:
            rws.append(rw.info())
        return api_response(200, 'Rows values', rws)
    else:
        return api_response(405, 'Method not allowed', 'This endpoint supports only a GET method.')