import pandas as pd
import json

from pymongo import MongoClient

client = MongoClient()

db = client['benchmark']

# the default collection 'evk' for energy
def_coll = db.evk

def add_dframe(dset):
   """
   extra feature to add: do not add duplicates
   """
   count = 0
   total = len(list(enumerate(dset.iterrows())))
   for n,(index,row) in enumerate(dset.iterrows()):
     doc = {key: row[key] for key in dset.columns}
     def_coll.insert(doc)
     count = count + 1
     #print ('inserting {0} {1}'.format(index,row['element'][index]))
   return ('inserted {0} of {1}'.format(count,total))

def find_by_query(query):
    """
    return a dataframe based on the query
    """
    docs = def_coll.find(query)
    dframe = pd.concat([pd.DataFrame({key: [doc[key]] for key in ['energy','volume']}) for doc in docs])
    return (dframe)

if __name__ == '__main__':
    #add_dframe(pd.read_csv('db_ready.csv'))
    dframe_ret = find_by_query({'element':'Ag','structure':'fcc','code':'VASP','k-point':64,'exchange':'PBE'})
    print (dframe_ret)
