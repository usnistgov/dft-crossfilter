# change sigma precs to sigma_E_kmin precs

import matplotlib
matplotlib.rcParams.update({'font.size': 18,'legend.fontsize':18, 'lines.markersize': 14})

import pandas as pd
import matplotlib.pyplot as plt



import numpy as np
import os
import sys
powerlaws_file = ''#'v4_PowerLawPairs_test_all.csv'

counter = 0

codes = 'VASP'
exchs = 'PBE'
# first crossfilter based on element, structure, property
def plot_all_together(name, a0, B, dB):
    """
    plot everything on one plot
    """

    plt.scatter(a0['x'], a0['y'], marker = ".", color='black', label=None)
    plt.scatter(B['x'], B['y'], marker = "*", color='blue', label=None)
    plt.scatter(dB['x'], dB['y'], marker = "v", color='red', label=None)

    try:
       ax = plt.gca()
       #ax.set_xscale('log')
       #ax.set_yscale('log')
       ax.set_xlabel('$\sigma_{E_0}$ in meV/atom')
       ax.set_ylabel("$\sigma_{V_0}$, $\sigma_{B}$, $\sigma_{B'}$ Precision in percent")
#       plt.plot(a0['x'], a0['f'], color='black', label='$\sigma_{V_0}$')
#       plt.plot(B['x'], B['f'], color='blue', label='$\sigma_{B}$')
#       plt.plot(dB['x'], dB['f'], color='red', label="$\sigma_{B'}$")
#       plt.xlim(min([min(a0['f']), min(B['f']), min(dB['f'])]), 10.0)
       plt.legend()
       plt.savefig('transformed_v6_'+name+'.pdf')
       plt.close()
    except:
        print ("plot problems")

def regress_with_R(name, X,Y,rscript='regression_linear_model.R'):
    """
    fits with R the variable X and Y according to
    the model provided by the script
    """

    dframe = pd.DataFrame({'X':X,'Y':Y})
    dframe.to_csv('Rdata.csv', index=False)
    #os.system('Rscript {}'.format(rscript))
    try:
       #print ('trying to execute Rscript {}'.format(rscript))
       os.system('Rscript {}'.format(rscript))
       #print ('executed')
    except:
       print (name, 'warnings')

    try:
       params = pd.read_csv('params.csv')
       preds = pd.read_csv('predicts.csv')
       plt.scatter(preds['x'], preds['y'])
       ax = plt.gca()
       ax.set_xscale('log')
       ax.set_yscale('log')
       plt.plot(preds['x'], preds['f'], color='red')
       plt.savefig(name)
       plt.close()

    except:
       params = pd.DataFrame( {'C':[np.nan],'M':[np.nan],'C_err':[np.nan],'M_err':[np.nan]} )

    return name, params['C'][0], params['C_err'][0], params['M'][0], params['M_err'][0], preds

def crossfilter_el_p(data,el,st,p):
    # first element
    el_data  = data[data['element']==el]
    st_el_data = el_data[el_data['structure']==st]
    #structs = np.unique(el_data['structure'])
    # perform crossfilter on first structure entry available, list others as well
    #print (structs)
    #st_el_data = el_data[el_data['structure']==structs[0]]
    # finally on property
    p_st_el_data = st_el_data[st_el_data['property']==p]
    kpoints_atom = list(p_st_el_data['k-point'])
    #kpoints_atom = [k**3 for k in kpoints]
    #print (el,st,p,kpoints_atom, list(p_st_el_data['perc_precisions']))
    if p=='E0':
      precs = [ abs(10*e) for e in p_st_el_data['perc_precisions'] ] 
    else:
      precs = list(p_st_el_data['perc_precisions'])
    return kpoints_atom, precs

if __name__=='__main__':
   """
   extract transforms
   """
   dataset = pd.read_csv('main_complete_perc_v6.csv')
   pbe_dataset = dataset[dataset['exchange']=='PBE']
   spin1_pbe_dataset = pbe_dataset[pbe_dataset['calculations_type']!='PseudoPot_ISPIN_2']
   spin1_pbe_dataset = spin1_pbe_dataset[spin1_pbe_dataset['calculations_type']!='PseudoPot_2_POT']
   spin1_pbe_dataset.to_csv('main_pbe_non_spin_v6.csv')
   focus_data = spin1_pbe_dataset[['element','structure','k-point','property', 'perc_precisions']]
   sorted_focus_data  = focus_data.sort_values(by=['property','k-point'])
   sorted_focus_data_no_outliers = sorted_focus_data[abs(sorted_focus_data['perc_precisions'])<50.0]
   mydata = sorted_focus_data_no_outliers
   mydata.to_csv('main_pbe_non_spin_v6_cleanedup.csv')
   sigma_Ps = {}
   kpoints_choices = []  
   st_el = [] 
   print ('finished sorted clean up.. crossfiltering..')
   transformed_set = {'element': [], 'structure': [], 'property':[], 'kpts_density': [], 'sigma_P_kmin':[]}
   prop_colors = { 'E0':'darkgreen', 'v0':'black', 'B':'blue', 'dB':'red' }
   prop_labels = { 'E0':'$\sigma_{E_0}kmin$', 'v0': '$\sigma_{V_0}kmin$', 'B': '$\sigma_Bkmin$', 'dB': "$\sigma_B'kmin$" }
   prop_markers = { 'E0':'.', 'v0': '*', 'B': '^', 'dB': "v" }
   for el in np.unique(mydata['element']):
      el_mydata  = mydata[mydata['element']==el]
      for st in np.unique(el_mydata['structure']):
         st_el_mydata = el_mydata[el_mydata['structure']==st]
         for p in ['E0','v0','B','dB']:#enumerate(np.unique(st_el_mydata['property'])):
             kpts, props_P = crossfilter_el_p(mydata, el, st, p) 
             #pd.DataFrame({'kpts':kpts,'props_P':props_P}).to_csv('check_max{}.csv'.format(counter))
             sigma_P_kmin = [ max(props_P[n:]) for n, k in enumerate(kpts) ]
             #print (len(sigma_P_kmin),len(kpts))
             name = '_'.join([st,el])
             #for n,pr in enumerate(props_P):
             #   print (name, ',', kpts[n], ',', p, ',', props_P[n], ',', sigma_P_kmin[n])
             #if p == 'E0':
             #   print (name, sigma_P_kmin)

             plt.scatter(kpts,sigma_P_kmin, label=prop_labels[p], color=prop_colors[p], marker=prop_markers[p]) 
             plt.plot(kpts,sigma_P_kmin, label=None, color=prop_colors[p], marker=prop_markers[p])
             sigma_Ps[p] = {'Sigma':sigma_P_kmin, 'Kpts':kpts}
             #else:
             #   plt.scatter(kpts,sigma_P_kmin, label=prop_labels[p], color=prop_colors[p], marker=prop_markers[p])
             
             #print (el,st,p)
             #print (props_P,sigma_P_kmin)

             for n,k in enumerate(kpts):

                transformed_set['element'].append(el)
                transformed_set['structure'].append(st)
                transformed_set['property'].append(p)
                transformed_set['kpts_density'].append(k)
                #print (k,sigma_P_kmin[n])
                transformed_set['sigma_P_kmin'].append(sigma_P_kmin[n])
             #trans_set = transformed_set
             #pd.DataFrame(trans_set).to_csv('trans{}set.csv'.format(i))

         plt.yscale('log') 
         plt.xscale('log') 
         plt.title(name) 
         plt.ylabel('Max $\sigma$ % for $k-$points choice')
         plt.xlabel('$k-$points density per atom')
         plt.legend()
         plt.tight_layout()
         plt.savefig('sigma_kmin_'+name+'.png')
         plt.close()
         print (el,st)
         for k in sigma_Ps.keys():
            sigma_indices = [n for n,s in enumerate(sigma_Ps[k]['Sigma']) if abs(float(s))<1.0]
            if sigma_indices:
               sigma_Ps[k].update( {'min_index':min(sigma_indices)} )
            else:
               sigma_Ps[k].update( {'min_index':1} )
               print ('Non precise warning for {0} {1}'.format(st, el))
         if (el,st)!=('Al','bcc'): 
            kpt_index = max([sigma_Ps[k]['min_index'] for k in sigma_Ps.keys()])
            print (np.log10(sigma_Ps['E0']['Kpts'][kpt_index]))
            kpoints_choices.append(np.log10(sigma_Ps['E0']['Kpts'][kpt_index]))
            st_el.append('_'.join([el,st]))
        
            
         #print (sigma_Ps)

         sigma_Ps = {}

         # k-points choices 
         #prop_sigmas  = [ n for n,s in enumerate(p[0]) for p in sigma_Ps if abs(float(s)) < 1.0 ]
         #kpt_sigmas = [ p[1] for p in sigma_Ps ]
#counter = counter + i
   print ('got the records for the new table..')
   transformed = pd.DataFrame(transformed_set)
   print ('made the dataframe and saving... finished. ')
   transformed.to_csv('v6_transformed_sigmas.csv')
   plt.close()
   pd.DataFrame({'Material':st_el, 'Kpoints_Density':kpoints_choices}).to_csv('Kpoints_Choices.csv')
   plt.hist(kpoints_choices)
   plt.savefig('Kpoints_Choices_histogram_logs.png')
   
   sys.exit()
   B_fits = {}
   v0_fits = {}
   dB_fits = {}
   for el in np.unique(transformed['element']):
      el_transformed  = transformed[transformed['element']==el]
      for st in np.unique(el_transformed['structure']):
         name = '_'.join([el,st])
         st_el_transformed = el_transformed[el_transformed['structure']==st]
         #for p in np.unique(st_el_transformed['property']):
         sigma_E0 = st_el_transformed[st_el_transformed['property']=='E0']
         X = list(abs(sigma_E0['sigma_P_kmin']))
         #print (len(X))
         #print (len(list(sigma_V0['sigma_P_kmin'])))
         #print (len(list(sigma_B['sigma_P_kmin'])))
         #print (len(list(sigma_dB['sigma_P_kmin'])))
         sigma_V0 = st_el_transformed[st_el_transformed['property']=='v0']
         sigma_B = st_el_transformed[st_el_transformed['property']=='B']
         sigma_dB = st_el_transformed[st_el_transformed['property']=='dB']
         l = min([len(X),len(list(sigma_V0['sigma_P_kmin'])), len(list(sigma_B['sigma_P_kmin'])), len(list(sigma_dB['sigma_P_kmin']))])
         #print (l)
         #print (len(list(sigma_dB['sigma_P_kmin'])))
         # perform power law fits - 3 and plot all together
         if l > 18:
           #print (name, l)
           v0_fits[name]= regress_with_R( name,X[:l],Y=list(abs(sigma_V0['sigma_P_kmin']))[:l], rscript='regression_linear_model.R' )
           B_fits[name]= regress_with_R( name,X[:l],Y=list(abs(sigma_B['sigma_P_kmin']))[:l], rscript='regression_linear_model.R' )
           dB_fits[name]= regress_with_R( name,X[:l],Y=list(abs(sigma_dB['sigma_P_kmin']))[:l], rscript='regression_linear_model.R' )

   #print (v0_fits, B_fits, dB_fits)
   #print ('performed fitting through')

   import matplotlib
   #print (matplotlib.rcParams.keys())
   matplotlib.rcParams.update({'font.size': 18,'legend.fontsize':16, 'lines.markersize': 14})

   #elem = 'fcc_Nb'
   #try:
   #  for e in B_fits.keys():
   #    elem_set = e, a0_fits[e][-1], B_fits[e][-1], dB_fits[e][-1]
   #    plot_all_together(*elem_set)
   #except:
   #  for e in dB_fits.keys():
   #    elem_set = e, a0_fits[e][-1], B_fits[e][-1], dB_fits[e][-1]
   #    plot_all_together(*elem_set)



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

   #meV_intercept = 0.001
   #E_intercept = np.log(meV_intercept)

   v0_intercept_c = [v0_fits[k][1] for k in v0_fits.keys()]#[ a0_fits[k](0) for k in a0_fits.keys() ]
   v0_intercept_c_err = [v0_fits[k][2] for k in v0_fits.keys()]
   #v0_intercept_x = [myLinFunc(E_intercept, v0_fits[k][3], v0_fits[k][1]) for k in v0_fits.keys()]

   B_intercept_c = [B_fits[k][1] for k in B_fits.keys()]#[ B_fits[k](0) for k in B_fits.keys() ]
   B_intercept_c_err = [B_fits[k][2] for k in B_fits.keys()]
   #B_intercept_x = [myLinFunc(E_intercept, B_fits[k][3], B_fits[k][1]) for k in B_fits.keys()]

   dB_intercept_c = [dB_fits[k][1] for k in dB_fits.keys()]#[ dB_fts[k](0) for k in dB_fits.keys() ]
   dB_intercept_c_err = [dB_fits[k][2] for k in dB_fits.keys()]
   #dB_intercept_x = [myLinFunc(E_intercept, dB_fits[k][3],dB_fits[k][1]) for k in dB_fits.keys()]

   limits_compare = [len(B_slope_m), len(dB_slope_m), len(v0_slope_m)]
   #print (limits_compare, min(limits_compare))
   l = min(limits_compare)
   ## saving the data on the slopes and intercepts

   DataSet = {'v0_names': v0_names[:l],
              'v0_M': v0_slope_m[:l],
              'v0_M_err':v0_slope_m_err[:l],
              'v0_C': v0_intercept_c[:l],
              'v0_C_err': v0_intercept_c_err[:l],
              #'v0_x': v0_intercept_x[:l],
              'B_names': B_names[:l],
              'B_M': B_slope_m[:l],
              'B_M_err': B_slope_m_err[:l],
              'B_C': B_intercept_c[:l],
              'B_C_err': B_intercept_c_err[:l],
              #'B_x': B_intercept_x[:l],
              'dB_names': dB_names[:l],
              'dB_M': dB_slope_m[:l],
              'dB_M_err': dB_slope_m_err[:l],
              'dB_C': dB_intercept_c[:l],
              'dB_C_err': dB_intercept_c_err[:l]}
              #'dB_x': dB_intercept_x[:l],}

   try:
     for e in list(B_fits.keys())[:l]:
       elem_set = e, v0_fits[e][-1], B_fits[e][-1], dB_fits[e][-1]
       plot_all_together(*elem_set)
   except:
     for e in list(dB_fits.keys())[:l]:
       elem_set = e, v0_fits[e][-1], B_fits[e][-1], dB_fits[e][-1]
       plot_all_together(*elem_set)

   plaws = pd.DataFrame(DataSet)
   #.to_csv('PowerLawPairs_test_2.csv',index=False)
   #           'B_E': B_extraps,
   #           'a0_E': a0_extraps,
   #           'dB_E': dB_extraps }

   ### plotting the histograms
   #fig,ax = plt.subplots(2,1)
   #print (type(v0_slope_m), type(B_slope_m), type(dB_slope_m))
   #n, bins, patches = plt.hist(np.array([v0_slope_m, B_slope_m, dB_slope_m]).transpose())
   #corrected_plaws = plaws.drop([6,31,11,10])

   #B_plaws = plaws[abs(plaws['B_C'])<5.0]
   #dB_plaws = B_plaws[abs(B_plaws['dB_C'])<5.0]
   #corrected_plaws = dB_plaws[abs(dB_plaws['v0_C']<6.0)]

   corrected_plaws = plaws

   #print (corrected_plaws)

   corrected_plaws.to_csv('transforms_v6_PowerLawPairs_test_all.csv')
   n, bins, patches = plt.hist(np.array([corrected_plaws['v0_M'], corrected_plaws['B_M'], corrected_plaws['dB_M']]).transpose())

   plt.setp(patches[0], color="black", label='$V_0$')
   plt.setp(patches[1], color="blue", label='$B$')
   plt.setp(patches[2], color="red", label="$B'$")

   plt.title("Sensitivity of Numerical Precision")
   #print (max([max(corrected_plaws['v0_M']), max(corrected_plaws['B_M']), max(corrected_plaws['dB_M'])]))
   #print (min([min(corrected_plaws['v0_M']), min(corrected_plaws['B_M']), min(corrected_plaws['dB_M'])]))
   plt.xlim(0.0, max([max(corrected_plaws['v0_M']), max(corrected_plaws['B_M']), max(corrected_plaws['dB_M'])]))
   plt.xlabel("Value of Slope $M$ (% per meV/atom)")
   plt.ylabel("Number of Elements")
   #plt.legend()

   plt.savefig('transforms_v6_{0}_{1}_slopes.pdf'.format(codes, exchs))
   plt.close()
   ## Intercepts
   #fig,ax = plt.subplots()

   #n, bins, patches = plt.hist(np.array([v0_intercept_c, B_intercept_c, dB_intercept_c]).transpose())
   n, bins, patches = plt.hist(np.array([corrected_plaws['v0_C'], corrected_plaws['B_C'], corrected_plaws['dB_C']]).transpose())

   plt.setp(patches[0], color="black")
   plt.setp(patches[1], color="blue")
   plt.setp(patches[2], color="red")
   plt.title("Numerical Precision at 1 meV/atom")

   minx = min([min(corrected_plaws['v0_C']), min(corrected_plaws['B_C']), min(corrected_plaws['dB_C'])])
   maxx = max([max(corrected_plaws['v0_C']), max(corrected_plaws['B_C']), max(corrected_plaws['dB_C'])])
   print(minx, maxx, range(int(minx)-1, int(maxx)+1,1))#[int(minx)+x for x in range(0,int(maxx)-int(minx))])
   labels  = ['$10^{'+str(s)+'}$' for s in range(int(minx)-1, int(maxx)+1,1)]#[int(minx)+x for x in range(0,int(maxx)-int(minx))]
   print (labels)
   ax = plt.gca()
   ax.set_xticklabels(labels)
#plt.xlim(int(minx),1.0)
   plt.xlim(int(minx)-1,int(maxx)+1)
   #print (labels)
   #plt.xlim(int(minx),1.0)
   #plt.xlim(-6,2)
   plt.xlabel("Value of Intercept $C$ (%)")
   plt.ylabel("Number of Elements")

   #plt.show()
   plt.tight_layout()
   plt.savefig('PPt_transforms_v6_{0}_{1}_intercepts.png'.format(codes, exchs))
   plt.close()


   #n, bins, patches = plt.hist(np.array([corrected_plaws['v0_x'], corrected_plaws['B_x'], corrected_plaws['dB_x']]).transpose())

   #plt.setp(patches[0], color="black")
   #plt.setp(patches[1], color="blue")
   #plt.setp(patches[2], color="red")
   #plt.title("Numerical Precision at {} meV/atom".format(str(meV_intercept)))
   #minx = min([min(corrected_plaws['v0_x']), min(corrected_plaws['B_x']), min(corrected_plaws['dB_x'])])
   #maxx = max([max(corrected_plaws['v0_x']), max(corrected_plaws['B_x']), max(corrected_plaws['dB_x'])])
   #print(minx, maxx)
   #labels  = ['$10^{'+str(s)+'}$' for s in [-6+x for x in range(0,8)]]
   #print (labels)
   #ax = plt.gca()
   #ax.set_xticklabels(labels)
   #plt.xlim(int(minx),1.0)
   #plt.xlim(-8,2)
   #plt.xlabel("Value of $\sigma_{V_0}, \sigma_{B_0}, \sigma_{B'}$ (%)")
   #plt.ylabel("Number of Elements")

   #plt.show()
   #plt.tight_layout()
   #print ('use saved {0}_{1}_intercepts_{2}.pdf'.format(codes, exchs, str(meV_intercept)))
   #plt.savefig('transform_v4_{0}_{1}_intercepts_{2}.pdf'.format(codes, exchs, str(meV_intercept)))
   #plt.close()


   #print('finished!')
