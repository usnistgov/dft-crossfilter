"""
submit data to a Flask db
use cases:
1.
"""

from flask import Flask

from flask_pymongo import PyMongo

import pandas as pd
import json

app = Flask(__name__)

app.config['MONGO_DBNAME'] = 'benchmark'


mongo = PyMongo(app)

dset = pd.read_csv('vasp_pbe_db_ready.csv')

@app.route('/add')
def add():
   evk = mongo.db.evk
   count = 0
   total = len(list(enumerate(dset.iterrows())))
   for n,(index,row) in enumerate(dset.iterrows()):
     doc = {key: row[key] for key in dset.columns}
     evk.insert(doc)
     count = count + 1
     #print ('inserting {0} {1}'.format(index,row['element'][index]))
   return ('inserted {0} of {1}'.format(count,total))

@app.route('/find')
def find():
   collect = mongo.db.evk
   docs = collect.find({'element':'Ag','structure':'fcc','code':'VASP','k-point':64,'exchange':'PBE'})
   dframe = pd.concat([pd.DataFrame({key: [doc[key]] for key in ['energy','volume']}) for doc in docs])
   return ('execute finished, you found', dframe.to_string())

if __name__ == '__main__':
    app.run(debug=True)
