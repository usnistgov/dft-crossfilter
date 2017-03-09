# script to crossfilter each the extrapolates file and the main data file tio
# create a file containing the precisions

## INPUTS
import pandas as pd
import numpy as np
import sys
import os
import getopt

main_file = pd.read_csv('MainCollection_v2moreclean.csv')
extrapolates = pd.read_csv('Pade_extrapolates_v2.csv') #pd.read_csv('Pade_w3_r1.csv')#pd.read_csv('Pade_final.csv')

def CLI(argv):
   inputfile = ''
   outputfile= ''
   try:
      opts, args = getopt.getopt(argv,"hi:o:",["ifile=", "ofile="])
      print (opts, args)
   except getopt.GetoptError:
      print ('check inputs')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print ('python kpoints_choices_plot.py -i <inputfile> -o <outputfile>')
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
      elif opt in ("-o"):
         outputfile = arg
   print ('Input file is "', inputfile)
   return inputfile, outputfile

def crossfilter(database, tags):
    """
    crossfilter based on the tags input
    """
    tag1_database = database[database['code']==tags['code']]
    tag2_database = tag1_database[tag1_database['exchange']==tags['exchange']]
    tag3_database = tag2_database[tag2_database['element']==tags['element']]
    tag4_database = tag3_database[tag3_database['structure']==tags['structure']]
    tag5_database = tag4_database[tag4_database['property']==tags['property']]

    return tag5_database

if __name__=='__main__':

    #extrapolates_file, out_file  = CLI(sys.argv[1:])
    #extrapolates = pd.read_csv(extrapolates_file)
    out_file = 'main_with_precs_v2.csv'
    withprecs = { }
    frames = []

    for c in np.unique(main_file['code']):
        code = main_file[main_file['code']==c]
        for e in np.unique(code['exchange']):
            exch = code[code['exchange']==e]
            for el in np.unique(exch['element']):
                elem = exch[exch['element']==el]
                for st in np.unique(elem['structure']):
                    struct = elem[elem['structure']==st]
                    for p in np.unique(struct['property']):
                        prop = struct[struct['property']==p]
                        val = prop['value']

                        tags = {'code': c,
                                'exchange': e,
                                'element': el,
                                'structure': st,
                                'property': p}

                        extdb = crossfilter(extrapolates, tags)
                        #print ("HERE")
                        #print (tags)
                        print (extdb.columns)

                        try:
                            #print ("AND HERE")
                            #print (extdb['extrapolate'])
                            try:
                               print (len(extdb['best_extrapolate']))
                               ext = float(extdb['best_extrapolate'])
                               err = float(extdb['best_error'])
                               order = float(extdb['best_order'])
                               print ("passed getting extrapolate")
                            except:
                               ext_list=[e.replace('[','').replace(']','').replace(' ','').split(',') for e in extdb['extrapolate']]
                               errors_list=[e.replace('[','').replace(']','').replace(' ','').split(',') for e in extdb['fit_error']]
                               print (ext_list, errors_list)
                               ext_floats = [float(e) for e in ext_list[0]]
                               err_floats = [float(e) for e in errors_list[0]]
                               print (len(ext_floats), len(err_floats))
                               err = min(err_floats)
                               order = float(extdb['best_order'])
                               ext = ext_floats[err_floats.index(min(err_floats))]
#                               ext = len(extdb['extrapolate'])#.replace('[','').replace(']','').split(',')
                               
                               # ext = extdb['extrapolate'][list(extdb['fit_error']).index(min(list(extdb['fit_error'])))]
                            #ext = ext[0]
                            #print (type(ext), len(ext), ext[0])
                            exts = [ ext for v in list(prop['value']) ]
                            errs = [ err for v in list(prop['value']) ]
                            orders = [ order for v in list(prop['value']) ]
                            prec = [ (abs(v - ext)/ext) * 100.0 for v in list(prop['value']) ]
                            prec_r = [ ((v - ext)/ext) * 100.0 for v in list(prop['value']) ]

                            withprecs['precision']  = prec_r
                            withprecs['perc_precisions'] = prec
                            withprecs['extrapolate'] = exts
                            withprecs['extrapolate_err'] = errs
                            withprecs['pade_order'] = orders

                            fins = {'code':[tags['code'] for i in prec],
                                    'exchange':[tags['exchange'] for e in prec],
                                    'element':[tags['element'] for i in prec],
                                    'structure':[tags['structure'] for i in prec],
                                    'property':[tags['property'] for i in prec],
                                    'k-point': list(prop['k-point']),
                                    'value': list(prop['value']),
                                    'value_error': list(prop['value_error']),
                                    'energy': list(prop['energy']),
                                    'volume': list(prop['volume']),
                                    'calculations_type': list(prop['calculations_type']),
                                    'bz_integration': list(prop['bz_integration'])}

                            withprecs.update(fins)

                            f = pd.DataFrame(withprecs)
                            frames.append(f)

                            withprecs = {} 


                        except:
                            print ('no extrapolate')

#                        if list(extdb['extrapolate'])[0]!='xxx':


    R = pd.concat(frames)
    #R.to_csv('PadePrecs_final_fix.csv')
    R.to_csv(out_file)
