import glob
from csv import loadCSV

def load(filename):
	filenames = []
	print glob.glob("benchmark/data/*")
	for f in glob.glob("benchmark/data/*"):
		if filename in f:
			filenames.append(f)
	selected = ""
	if len(filenames) > 1:
		print "Warning: Many files have the same name! ("+",".join(filenames)+")."
		print "Warning: Only the first one will be loaded."
		selected = filenames[0]
	elif len(filenames) == 0:
		print "Error: No file with that name found."
		return None
	else:
		selected = filenames[0]

	name = selected.split(".")[0]
	extension = selected.split(".")[1]

	if 'csv' in extension:
		return loadCSV(name)

