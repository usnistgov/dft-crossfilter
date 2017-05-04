import pandas as pd
import json

from pymongo import MongoClient

client = MongoClient()

db = client['benchmark']

def_coll = db.evk

def find_by_query(query):
    """
    return a dataframe based on the query
    """
    docs = def_coll.find(query)
    dframe = pd.concat([pd.DataFrame({key: [doc[key]] for key in ['energy','volume']}) for doc in docs])
    return (dframe)

if __name__ == '__main__':
    dframe_ret = find_by_query({'element':'Ag','structure':'fcc','code':'VASP','k-point':64,'exchange':'PBE'})
    print (dframe_ret)
