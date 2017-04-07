"""
complete Pade analysis workflow
from E-V fitted data to
the precision data, plots and
histograms
GOAL: to be the one script that is used by
the database web app to be activated in part
on sensibly crossfiltered data
and also on the a complete new set of data
in an object oriented way.
"""
# must be executed within
# an anaconda shell with R-3.3.1
# this script will command the required R scripts
# 1. first crossfilter according to the project
# 2. perform the Pade analysis on each of the crossfiltered parts
# 3. calculate the precisions and create the database ready data with precisions
# 4. perform analysis on the precision relationship between properties
# 5. describe the analysis with histograms

import os
import pandas as pd
import numpy as np
from glob import glob

class DatabaseData:
    """
    a class that contains all the attributes of
    the DatabaseData and builds the database
    to a complete set of its endpoints
    """
    def __init__(self, orig_data, other_data_attributes=None):
        """
        constructor function that builds from original data
        of values of the different properties. simply constructs
        data from a k-point, value attribute csv file with pandas
        """
        self.orig_data = orig_data
        self.other_data_attributes = other_data_attributes
        # optionally check for the endpoints and then construct
        self.data = pd.read_csv(self.orig_data)

    def create_crossfilts(self,tags):
        """
        function to create crossfilters by a specified tag
        this function can serve further as a
        generalized query to the mongoDB
        """
        tag1_database = self.pade_analysis_table[self.pade_analysis_table['code']==tags['code']]
        tag2_database = tag1_database[tag1_database['exchange']==tags['exchange']]
        tag3_database = tag2_database[tag2_database['element']==tags['element']]
        tag4_database = tag3_database[tag3_database['structure']==tags['structure']]
        tag5_database = tag4_database[tag4_database['property']==tags['property']]
        #print (tag5_database)
        return tag5_database

    def auto_crossfilter(self):
        """
        automatic crossfiltering into a folder
        'Crossfilts' for a new dataset for which
        the precisions need to be computed
        """
        if not os.path.isdir('AutoCrossfilts'):
            os.mkdir('AutoCrossfilts')
        names = []
        tags = []
        codes = np.unique(self.data['code'])
        for c in codes:
            code = self.data[self.data['code']==c]
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

                             prop_error = list(pr_el_ex_struct_code['value_error'])
                             prop = list(pr_el_ex_struct_code['value'])
                             prop_err = list(pr_el_ex_struct_code['value_error'])
                             kpts = list(pr_el_ex_struct_code['k-point'])
                             # convert everything into kpts_density
                             k_atom = kpts #[ k**3 for k in kpts ]
                             Pade_df = pd.DataFrame({'Kpts_atom': k_atom, 'P': prop, 'P_err': prop_err})
                             TAG =   {'element':el,
                                      'structure':struct,
                                      'exchange':ex,
                                      'code':c,
                                      'property':pr}
                             NAME = '_'.join([pr, el, ex, struct, c])+'.csv'
                             names.append(NAME)
                             tags.append(TAG)
                             print ("Writing {} ..".format(NAME))
                             Pade_df.to_csv('AutoCrossfilts'+os.sep+NAME, index=False)

        self.crossfilt_names = names
        self.tags = tags

    def read_crossfilts(self, filename):
            """
            file reader for crossfilts
            """
            self.crossfilt = pd.read_csv(filename)

    def run_pade_through_R(self, rscript):
            """
            Runs the Pade analysis through a supplied R script
            runs the Pade through a python subprocess call to nls.R
            on the input crossfilt
            - copies the input to Rdata.csv for input to nls.R
            - retrieves the output of nls.R that is pasted out into csv file
              that can be read back into pandas
              .. element, structure, exchange, code, property, extrapolate, fit error
              which can serve as another reference collection for calculation of
              the precision from the main database.
            """
            records = []
            for tag in self.tags:
                result = {'element':tag['element'],
                'structure':tag['structure'],
                'exchange':tag['exchange'],
                'code':tag['code'],
                'property':tag['property']}
                NAME = '_'.join([result['property'], result['element'], \
                                 result['exchange'], result['structure'], result['code']])+'.csv'
                os.system('cp AutoCrossfilts/{} Rdata.csv'.format(NAME))
                print ('copied {}'.format(NAME))

                try:
                    os.system('Rscript {}'.format(rscript))
                    print ('R executed')
                    R_result = pd.read_csv('Result.csv')
                    key = list(R_result['Error']).index(min(list(R_result['Error'])))
                    result['extrapolate'] = list(R_result['Extrapolate'])
                    result['best_extrapolate'] = list(R_result['Extrapolate'])[key]
                    result['best_error'] = list(R_result['Error'])[key]
                    result['best_order'] = list(R_result['Order'])[key]
                    result['fit_error'] = list(R_result['Error'])
                    result['pade_order'] = list(R_result['Order'])
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

                records.append(result)

            self.pade_analysis_table =\
                 pd.DataFrame({'element': [r['element'] for r in records],
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
            self.pade_analysis_table.to_csv('Pade_Extrapolates.csv', index=False)

    def create_precisions(self):
        """
        reads through the original data collection by crossfilted record,
        matches the crossfilt with the extrapolates crossfilt
        and appends the precision column to the crossfiltered record
        """
        self.pade_analysis_table = pd.read_csv('Pade_Extrapolates.csv')
        frames = []
        withprecs = {}
        for c in np.unique(self.data['code']):
            code = self.data[self.data['code']==c]
            for e in np.unique(code['exchange']):
                exch = code[code['exchange']==e]
                for el in np.unique(exch['element']):
                    elem = exch[exch['element']==el]
                    #print ('at el')
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
                            extdb = self.create_crossfilts(tags)

                            try:
                                try:
                                    #print (len(extdb['best_extrapolate']))
                                    ext = float(np.unique(extdb['best_extrapolate'])[0])
                                    err = float(np.unique(extdb['best_error'])[0])
                                    order = float(np.unique(extdb['best_order'])[0])
                                except:
                                    #print (extdb['extrapolate'])
                                    ext_list=[e.replace('[','').replace(']','').replace(' ','').split(',') for e in extdb['extrapolate']]
                                    errors_list=[e.replace('[','').replace(']','').replace(' ','').split(',') for e in extdb['fit_error']]
                                    print (ext_list, errors_list)
                                    ext_floats = [float(e) for e in ext_list[0]]
                                    err_floats = [float(e) for e in errors_list[0]]
                                    print (len(ext_floats), len(err_floats))
                                    err = min(err_floats)
                                    order = float(extdb['best_order'])
                                    ext = ext_floats[err_floats.index(min(err_floats))]
                                exts = [ ext for v in list(prop['value']) ]
                                errs = [ err for v in list(prop['value']) ]
                                orders = [ order for v in list(prop['value']) ]
                                prec = [ abs((float(v) - ext)/ext) * 100.0 for v in list(prop['value']) ]
                                if min(prec)>10:
                                    print ('More than 10', tags['element'],tags['structure'], tags['property'], len(prec), min(prec))
                                else:
                                    print ('Good precs', tags['element'],tags['structure'], tags['property'], len(prec) )
                                    prec_r = [ ((float(v) - ext)/ext) * 100.0 for v in list(prop['value']) ]
                                    withprecs['precision']  = prec_r
                                    withprecs['perc_precisions'] = prec
                                    withprecs['extrapolate'] = exts
                                    withprecs['extrapolate_err'] = errs
                                    withprecs['pade_order'] = orders
                                    #print ('MADE IT THRU precs')
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
        self.data = pd.concat(frames)

    def linear_regress_with_R(self):
        """
        linear regression with R
        """
        dframe = pd.DataFrame({'X':X,'Y':Y})
        dframe.to_csv('Rdata.csv', index=False)
        try:
           os.system('Rscript {}'.format(rscript))
        except:
           print (name, 'warnings')
        try:
           params = pd.read_csv('params.csv')
           preds = pd.read_csv('predicts.csv')
           #plt.scatter(preds['x'], preds['y'])
           #ax = plt.gca()
           #ax.set_xscale('log')
           #ax.set_yscale('log')
           #plt.plot(preds['x'], preds['f'], color='red')
           #plt.savefig(name)
           #plt.close()
        except:
           params = pd.DataFrame( {'C':[np.nan],'M':[np.nan],'C_err':[np.nan],'M_err':[np.nan]} )

        print ('{0},{1},{2},{3},{4}'.format(name.split('_')[:-2], params['C'][0], params['C_err'][0], params['M'][0], params['M_err'][0]))
        return name, params['C'][0], params['C_err'][0], params['M'][0], params['M_err'][0], preds

    def inter_property_power_law_analysis(self,codes,exchs):
        """
        regression of power laws on the crossfilts
        """

        code = self.data[self.data['code']==codes]
        exch_code = code[code['exchange']==exchs]
        # variable definitions for fits and scatter data
        dB_fits = { }
        v0_fits = { }
        B_fits = { }
        # separate data by element first and then by structure to rename
        # the data labels as element_structure
        for el in np.unique(exch_code['element']):
           el_exch_code = exch_code[exch_code['element']==el]
           for st in np.unique(el_exch_code['structure']):
                st_el_exch_code = el_exch_code[el_exch_code['structure']==st]

                E_st_el_exch_code = st_el_exch_code[st_el_exch_code['property']=='E0']  ## VASP
                dB_st_el_exch_code = st_el_exch_code[st_el_exch_code['property']=='dB']
                B_st_el_exch_code = st_el_exch_code[st_el_exch_code['property']=='B']
                v0_st_el_exch_code = st_el_exch_code[st_el_exch_code['property']=='v0']

                l = min([len(E_st_el_exch_code['perc_precisions']),len(v0_st_el_exch_code['perc_precisions']),\
                       len(B_st_el_exch_code['perc_precisions']), len(dB_st_el_exch_code['perc_precisions'])])
                print (st,el)
                if l > 18:
                   name = '_'.join([st,el])
                   # perform fit on data log(abs(perc_precision))
                   #print ('fitting dB for {}'.format(name))
                   X = list(10*abs(E_st_el_exch_code['perc_precisions']))[:l]
                   Y = list(abs(dB_st_el_exch_code['perc_precisions']))[:l]
                   names = '_'.join(['R_dB',name])
                   params_powR = self.linear_regress_with_R(names,X,Y)
                   dB_fits['_'.join([st,el])]=params_powR

                   X = list(10*abs(E_st_el_exch_code['perc_precisions']))[:l]
                   Y = list(abs(v0_st_el_exch_code['perc_precisions']))[:l]
                   names = '_'.join(['R_v0',name])
                   params_powR = regress_with_R(names, X,Y)
                   #print ('TESTING With R')
                   #print('{0} v0 yields C= {1} +- {2} and M= {3} +- {4}'.format(*params_powR))
                   #print ('v0, {0}, {1}, {2}, {3}'.format(name, params_powR[1:5]))
                   v0_fits['_'.join([st,el])]=params_powR
                   # perform fit on data log(abs(perc_precision))
                   #print ('fitting B for {}'.format(name))
                   X = list(10*abs(E_st_el_exch_code['perc_precisions']))[:l]
                   Y = list(abs(B_st_el_exch_code['perc_precisions']))[:l]
                   names = '_'.join(['R_B',name])
                   params_powR = regress_with_R(names,X,Y)
                   #print ('TESTING With R')
                   #print('{0} B yields C= {1} +- {2} and M= {3} +- {4}'.format(name, *params_powR))
                   #print ('B, {0}, {1}, {2}, {3}'.format(name, params_powR[1:5]))
                   B_fits['_'.join([st,el])]=params_powR

        try:
          for e in B_fits.keys():
            elem_set = e, v0_fits[e][-1], B_fits[e][-1], dB_fits[e][-1]
            plot_all_together(*elem_set)
        except:
          for e in dB_fits.keys():
            elem_set = e, v0_fits[e][-1], B_fits[e][-1], dB_fits[e][-1]
            plot_all_together(*elem_set)

        print (len(B_fits.keys()), len(dB_fits.keys()), len(v0_fits.keys()))

        B_slope_m = [B_fits[k][3] for k in B_fits.keys()]#[ B_fits[k](1) - B_fits[k](0) for k in B_fits.keys() ]
        B_slope_m_err = [B_fits[k][4] for k in B_fits.keys()]
        #B_slope_eqn = [myLinFunc(B_fits[k][3], B_fits[k][1]) for k in B_fits.keys()]
        B_names = [B_fits[k][0] for k in B_fits.keys()]

        v0_slope_m = [v0_fits[k][3] for k in v0_fits.keys()] #[ a0_fits[k](1) - a0_fits[k](0) for k in a0_fits.keys() ]
        v0_slope_m_err = [v0_fits[k][4] for k in v0_fits.keys()]
        v0_names = [v0_fits[k][0] for k in v0_fits.keys()]

        dB_slope_m = [dB_fits[k][3] for k in dB_fits.keys()]#[ dB_fits[k](1) - dB_fits[k](0) for k in dB_fits.keys() ]
        dB_slope_m_err = [dB_fits[k][4] for k in dB_fits.keys()]
        dB_names = [dB_fits[k][0] for k in dB_fits.keys()]

        meV_intercept = 0.001
        E_intercept = np.log(meV_intercept)

        v0_intercept_c = [v0_fits[k][1] for k in v0_fits.keys()]#[ a0_fits[k](0) for k in a0_fits.keys() ]
        v0_intercept_c_err = [v0_fits[k][2] for k in v0_fits.keys()]
        v0_intercept_x = [myLinFunc(E_intercept, v0_fits[k][3], v0_fits[k][1]) for k in v0_fits.keys()]

        B_intercept_c = [B_fits[k][1] for k in B_fits.keys()]#[ B_fits[k](0) for k in B_fits.keys() ]
        B_intercept_c_err = [B_fits[k][2] for k in B_fits.keys()]
        B_intercept_x = [myLinFunc(E_intercept, B_fits[k][3], B_fits[k][1]) for k in B_fits.keys()]

        dB_intercept_c = [dB_fits[k][1] for k in dB_fits.keys()]#[ dB_fts[k](0) for k in dB_fits.keys() ]
        dB_intercept_c_err = [dB_fits[k][2] for k in dB_fits.keys()]
        dB_intercept_x = [myLinFunc(E_intercept, dB_fits[k][3],dB_fits[k][1]) for k in dB_fits.keys()]

        limits_compare = [len(B_slope_m), len(dB_slope_m), len(v0_slope_m)]
        print (limits_compare, min(limits_compare))
        l = min(limits_compare)
        ## saving the data on the slopes and intercepts

        DataSet = {'v0_names': v0_names[:l],
                   'v0_M': v0_slope_m[:l],
                   'v0_M_err':v0_slope_m_err[:l],
                   'v0_C': v0_intercept_c[:l],
                   'v0_C_err': v0_intercept_c_err[:l],
                   'v0_x': v0_intercept_x[:l],
                   'B_names': B_names[:l],
                   'B_M': B_slope_m[:l],
                   'B_M_err': B_slope_m_err[:l],
                   'B_C': B_intercept_c[:l],
                   'B_C_err': B_intercept_c_err[:l],
                   'B_x': B_intercept_x[:l],
                   'dB_names': dB_names[:l],
                   'dB_M': dB_slope_m[:l],
                   'dB_M_err': dB_slope_m_err[:l],
                   'dB_C': dB_intercept_c[:l],
                   'dB_C_err': dB_intercept_c_err[:l],
                   'dB_x': dB_intercept_x[:l],}

        #try:
        #  for e in list(B_fits.keys())[:l]:
        #    elem_set = e, v0_fits[e][-1], B_fits[e][-1], dB_fits[e][-1]
        #    plot_all_together(*elem_set)
        #except:
        #  for e in list(dB_fits.keys())[:l]:
        #    elem_set = e, v0_fits[e][-1], B_fits[e][-1], dB_fits[e][-1]
        #    plot_all_together(*elem_set)

        self.plaws = pd.DataFrame(DataSet)

    def plot_histograms(self, codes, exchs):
        """
        histogram plotter
        """
        n, bins, patches = plt.hist(np.array([self.plaws['v0_M'], self.plaws['B_M'], self.plaws['dB_M']]).transpose())

        plt.setp(patches[0], color="black", label='$a_0$')
        plt.setp(patches[1], color="blue", label='$B$')
        plt.setp(patches[2], color="red", label="$B'$")

        plt.title("Sensitivity of Numerical Precision")
        #print (max([max(self.plaws['v0_M']), max(self.plaws['B_M']), max(self.plaws['dB_M'])]))
        #print (min([min(self.plaws['v0_M']), min(self.plaws['B_M']), min(self.plaws['dB_M'])]))
        plt.xlim(0.0, max([max(self.plaws['v0_M']), max(self.plaws['B_M']), max(self.plaws['dB_M'])]))
        plt.xlabel("Value of Slope $M$ (% per meV/atom)")
        plt.ylabel("Number of Elements")
        #plt.legend()

        plt.savefig('v6_{0}_{1}_slopes.pdf'.format(codes, exchs))
        plt.savefig('PPt_v6_{0}_{1}_slopes.png'.format(codes, exchs))
        plt.close()
        ## Intercepts
        #fig,ax = plt.subplots()

        #n, bins, patches = plt.hist(np.array([v0_intercept_c, B_intercept_c, dB_intercept_c]).transpose())
        n, bins, patches = plt.hist(np.array([self.plaws['v0_C'], self.plaws['B_C'], self.plaws['dB_C']]).transpose())

        plt.setp(patches[0], color="black")
        plt.setp(patches[1], color="blue")
        plt.setp(patches[2], color="red")
        plt.title("Numerical Precision at 1 meV/atom")
        minx = min([min(self.plaws['v0_C']), min(self.plaws['B_C']), min(self.plaws['dB_C'])])
        maxx = max([max(self.plaws['v0_C']), max(self.plaws['B_C']), max(self.plaws['dB_C'])])
        print(minx, maxx)
        labels  = ['$10^{'+str(s)+'}$' for s in [-6+x for x in range(0,8)]]
        #print (labels)
        ax = plt.gca()
        ax.set_xticklabels(labels)
        #plt.xlim(int(minx),1.0)
        plt.xlim(-6,2)
        plt.xlabel("Value of Intercept $C$ (%)")
        plt.ylabel("Number of Elements")

        #plt.show()
        plt.tight_layout()
        plt.savefig('{0}_{1}_intercepts.pdf'.format(codes, exchs))
        plt.savefig('PPt_{0}_{1}_intercepts.png'.format(codes, exchs))
        plt.close()


        n, bins, patches = plt.hist(np.array([self.plaws['v0_x'], self.plaws['B_x'], self.plaws['dB_x']]).transpose())

        plt.setp(patches[0], color="black")
        plt.setp(patches[1], color="blue")
        plt.setp(patches[2], color="red")
        plt.title("Numerical Precision at {} meV/atom".format(str(meV_intercept)))
        minx = min([min(self.plaws['v0_x']), min(self.plaws['B_x']), min(self.plaws['dB_x'])])
        maxx = max([max(self.plaws['v0_x']), max(self.plaws['B_x']), max(self.plaws['dB_x'])])
        print(minx, maxx, range(int(minx)-1, int(maxx)+1,1))#[int(minx)+x for x in range(0,int(maxx)-int(minx))])
        labels  = ['$10^{'+str(s)+'}$' for s in range(int(minx)-1, int(maxx)+1,1)]#[int(minx)+x for x in range(0,int(maxx)-int(minx))]]
        print (labels)
        ax = plt.gca()
        ax.set_xticklabels(labels)
        #plt.xlim(int(minx),1.0)
        plt.xlim(int(minx)-1,int(maxx)+1)
        plt.xlabel("Value of $\sigma_{V_0}, \sigma_{B_0}, \sigma_{B'}$ (%)")
        plt.ylabel("Number of Elements")

        #plt.show()
        plt.tight_layout()
        print ('use saved LDA_{0}_{1}_intercepts_{2}.pdf'.format(codes, exchs, str(meV_intercept)))
        plt.savefig('LDA_{0}_{1}_intercepts_{2}.pdf'.format(codes, exchs, str(meV_intercept)))
        plt.savefig('PPt_LDA_{0}_{1}_intercepts_{2}.png'.format(codes, exchs, str(meV_intercept)))
        plt.close()
        pass


if __name__ == '__main__':
    """
    test
    """
    data_tables = []
    pade_tables = []
    for f in ['bcc_bcc_rerun_real_data.csv']:
       dataSet = DatabaseData(orig_data='bcc_bcc_reruns_data.csv')
       dataSet.auto_crossfilter()
       dataSet.run_pade_through_R(rscript='hennig_nls.R')
       dataSet.create_precisions()
       #data_tables.append(dataSet.data)
       dataSet.data.to_csv(f+'Table.csv', index=False)
       #pade_tables.append(dataSet.pade_analysis_table)
       dataSet.pade_analysis_table.to_csv(f+'Pade.csv', index=False)
    #pd.concat(data_tables).to_csv('complete_table.csv')
    #pd.concat(pade_tables).to_csv('complete_pade.csv')
