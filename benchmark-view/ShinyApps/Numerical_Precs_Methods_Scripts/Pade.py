## python script to crossfilter out and run Pade approximation on a database 
## inputs : Pade script nls.R, 
##          and one of database source csv or path to crossfiltered named
##          files            
## the necessary details to fit a Pade approximation for
## variety of functions.

import pandas as pd
import numpy as np
import os
from glob import glob
import sys

def crossfilters(database):
    """
    crossfilter out completely a collection
    """
    database = database #pd.read_csv('MainCollection.csv')

    # crossfilter down to VASP's fcc Nb Bulk modulus

    names = []

    codes = np.unique(database['code'])    

    for c in codes:
        code = database[database['code']==c]
        structures = np.unique(code['structure'])
        for struct in structures:
            struct_code = code[code['structure']==struct]
            exchanges = np.unique(struct_code['exchange'])
            for ex in exchanges:
                ex_struct_code = struct_code[struct_code['exchange']==ex]
                elements = np.unique(ex_struct_code['element'])
                for el in elements:
                     el_ex_struct_code = ex_struct_code[ex_struct_code['element']==el]
                     properties = el_ex_struct_code['property']
                     for pr in properties:
                         pr_el_ex_struct_code = el_ex_struct_code[el_ex_struct_code['property']==pr]

                         prop = list(pr_el_ex_struct_code['value'])
                         kpts = list(pr_el_ex_struct_code['k-point'])

                         k_atom = [ k**3 for k in kpts ]

                         Pade_df = pd.DataFrame({'Kpts_atom': k_atom, 'P': prop})

                         TAG =   {'element':el,
                                  'structure':struct,
                                  'exchange':ex,
                                  'code':c,
                                  'property':pr}

                         NAME = '_'.join([pr, el, ex, struct, c])+'.csv'
                         names.append( (NAME,TAG) )
                         print ("Writing {} ..".format(NAME))
                         Pade_df.to_csv('Crossfilts/'+NAME, index=False)

    return names


def read_crossfilts_from_file(filename):
    """
    reads the crossfiltered file and also decomposes the filename 
    into the tags and sends the crossfilt and the tags 
    """

    if len(filename[11:-4].split('_')) == 6:
        pr, el, ex, _,  struct, c = filename[11:-4].split('_')
        ex = '_'.join([ex,_])
    else:
        pr, el, ex, struct, c = filename[11:-4].split('_')
   
    tags = {'element': el,
            'property': pr,
            'exchange': ex, 
            'code': c,
            'structure':struct}
    return filename, tags

def run_pade_through_R(rscript, crossfilt, tags):
    """
    runs the Pade through a python subprocess call to nls.R
    on the input crossfilt
    - copies the input to Rdata.csv for input to nls.R
    - retrieves the output of nls.R that is pasted out into csv file
      that can be read back into pandas
      .. element, structure, exchange, code, property, extrapolate, fit error
      which can serve as another reference collection for calculation of
      the precision from the main database.
    """

    result = {'element':tags['element'],
              'structure':tags['structure'],
              'exchange':tags['exchange'],
              'code':tags['code'],
              'property':tags['property']}

    os.system('cp {} Rdata.csv'.format(crossfilt))
    # for making the first database 
    # os.system('cp Crossfilts/{} Rdata.csv'.format(crossfilt))
    # os.mkdir(crossfilt) 
    #os.chdir(crossfilt)   
    #os.system('cp ../{} Rdata.csv'.format(crossfilt))
    #os.system('cp ../{0} {0}'.format(rscript))

    print ('copied {}'.format(crossfilt))

    try:
       os.system('Rscript {}'.format(rscript))
       print ('R executed')
       R_result = pd.read_csv('Result.csv')
       key = list(R_result['Error']).index(min(list(R_result['Error'])))
       result['extrapolate'] = list(R_result['Extrapolate'])#[key]
       result['best_extrapolate'] = list(R_result['Extrapolate'])[key]
       result['best_error'] = list(R_result['Error'])[key]
       result['best_order'] = list(R_result['Order'])[key]
       result['fit_error'] = list(R_result['Error'])#[key]
       result['pade_order'] = list(R_result['Order'])#[key]
       #result['precision'] = list(R_result['Precisions'])
       print ("R success")

    except:
       print ("R failure")
       result['best_extrapolate'] = 'xxx'
       result['best_error'] = 'xxx'
       result['best_order'] = 'xxx'
       result['extrapolate'] = 'xxx'
       result['fit_error'] = 'xxx'
       result['pade_order'] = 'xxx'
    
    # os.chdir('../')
    #print (result, type(result))
    #pade_result = pd.DataFrame(result)

    return result



if __name__=='__main__':
    """
    calculate the fit for a given crossfiltered set
    for different Pade sets

    first Milestone - one crossfiltered set :
     Nb B for m+n orders (m, n =2-4)  .. output file Pade.csv

    
    """

    #database_path = 'MainCollection_v2moreclean.csv'

    rscript = 'hennig_nls.R'#'nls_kpts_choices.R'
    database_path = None
    crossfilts_path = 'Crossfilts/*.csv'
    #crossfilts_path = None

    output_filename = 'Pade_extrapolates_v2.csv'#'Pade_kpts_choices_leave3_10.csv'

    if database_path:
        print ("Performing crossfiltering on {}..".format(database_path))
        filetags = crossfilters(pd.read_csv(database_path))
    elif crossfilts_path:
        print ("Reading crossfilters from {}..".format(crossfilts_path))
        filetags = [read_crossfilts_from_file(f) for f in glob(crossfilts_path) ]
        length_crossfilts = len(filetags)
    else:
        print ('input not provided')
        sys.exit(0)

    records = []

    print ("Running Pade..")

    for n, (f,t) in enumerate(filetags):
        print ("Running through {0} of {1}".format(n, length_crossfilts))
        records.append( run_pade_through_R(rscript, f, t) )

    Pade_analysis= pd.DataFrame({'element': [r['element'] for r in records], 
                  'structure': [r['structure'] for r in records],
                  'exchange': [r['exchange'] for r in records],
                  'code': [r['code'] for r in records],
                  'property': [r['property'] for r in records],
                  'best_extrapolate': [r['best_extrapolate'] for r in records],
                  'best_error': [r['best_error'] for r in records], 
                  'best_order': [r['best_order'] for r in records],
                  'extrapolate': [r['extrapolate'] for r in records],
                  'fit_error': [r['fit_error'] for r in records],
                  'pade_order': [r['pade_order'] for r in records]  })

#    pade_analysis = pd.concat(records)

    # remove the index and duplicates 

    print ("Writing out Pade analysis... ")

    Pade_analysis.to_csv(output_filename)

    Pade_analysis = pd.read_csv(output_filename)

    del Pade_analysis['Unnamed: 0']

    Pade_analysis.drop_duplicates(inplace=True)

    Pade_analysis.to_csv(output_filename)







