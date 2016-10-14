from __future__ import absolute_import

from os.path import dirname, join

def loadCSV(filename):

	try:
	    import pandas as pd
	except ImportError as e:
	    raise RuntimeError("load data requires pandas (http://pandas.pydata.org) to be installed")

	data = pd.read_csv('%s.csv'%filename)
	return data