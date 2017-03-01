import matplotlib.pyplot as plt
import os
#gaussian_numbers = np.random.randn(1000)
# slope m and intercept a
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

def myPowFunc(x, a, b):
    return a * np.power(x, b)

def myExpFunc(x,a,b):
    return a*np.exp(b*x)

def myLinFunc(x,a,b):
    return a*x+b

def plot_all_together(name, a0, B, dB):
    """
    plot everything on one plot
    """

    plt.scatter(a0['x'], a0['y'], marker = ".", color='black', label=None)
    plt.scatter(B['x'], B['y'], marker = "*", color='blue', label=None)
    plt.scatter(dB['x'], dB['y'], marker = "v", color='red', label=None)

    try:
       ax = plt.gca()
       ax.set_xscale('log')
       ax.set_yscale('log')
       ax.set_xlabel('\sigma_{E} in meV/atom')
       ax.set_ylabel("$\sigma_{V_0}$, $\sigma_{B}$, $\sigma_{B'}$ Precision in percent")
       plt.plot(a0['x'], a0['f'], color='black', label='$\sigma_{V_0}$')
       plt.plot(B['x'], B['f'], color='blue', label='$\sigma_{B}$')
       plt.plot(dB['x'], dB['f'], color='red', label="$\sigma_{B'}$")
       plt.xlim(min([min(a0['f']), min(B['f']), min(dB['f'])]), 10.0)
       plt.legend()
       plt.savefig('v2_'+name+'.pdf')
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
    try:
       os.system('Rscript {}'.format(rscript))
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


def evalFit(X,Y,name, fit_type='loglog', p='dB'):
    """
    examine if the fit makes sense
    """
    plt.scatter(X,Y)

    try:

       if fit_type=='loglog':

           popt, pcov = curve_fit(myPowFunc, X, Y)
           newX = np.logspace(-6, 1, base=10)

           plt.plot(newX, myPowFunc(newX, *popt), 'r-',\
                        label="({0:.3f}*x**{1:.3f})".format(*popt))
           err = np.sqrt(np.diag(pcov))
           ax=plt.gca()
           ax.set_xscale('log')
           ax.set_yscale('log')
           plt.savefig('_'.join([fit_type,p,name]))
           plt.close()

           #print ("\tc = popt[0] = {0}\n\tm = popt[1] = {1}".format(*popt))
           #print("\tE_c = pcov[0] = {0}\n\tE_m = pcov[1] = {1}\n\tE_c = pcov[2] = {0}\n\tE_m = pcov[3] = {1}".format(*pcov))
           #print("Std.dev error is {}".format(np.sqrt(np.diag(pcov))))
           return name, fit_type, popt, err

       elif fit_type=='loglin':

           popt, pcov = curve_fit(myExpFunc, X, Y)
           newX = np.logspace(-6, 1, base=10)

           plt.plot(newX, myPowFunc(newX, *popt), 'r-',\
                        label="({0:.3f}*x**{1:.3f})".format(*popt))

           ax=plt.gca()
           ax.set_xscale('log')
           #ax.set_yscale('log')

           err = np.sqrt(np.diag(pcov))
           #print ("\tc = popt[0] = {0}\n\tm = popt[1] = {1}".format(*popt))
           #print("\tE_c = pcov[0] = {0}\n\tE_m = pcov[1] = {1}\n\tE_c = pcov[2] = {0}\n\tE_m = pcov[3] = {1}".format(*pcov))
           #print("Std.dev error is {}".format(np.sqrt(np.diag(pcov))))
           plt.savefig('_'.join([fit_type,p,name]))
           plt.close()

           return name, fit_type, popt, err

       elif fit_type=='linlin':

           popt, pcov = curve_fit(myLinFunc, X, Y)
           newX = np.linspace(0.000001,10)

           plt.plot(newX, myLinFunc(newX, *popt), 'r-',\
                        label="({0:.3f}*x**{1:.3f})".format(*popt))

           #print ("\tc = popt[0] = {0}\n\tm = popt[1] = {1}".format(*popt))
           #print("\tE_c = pcov[0] = {0}\n\tE_m = pcov[1] = {1}\n\tE_c = pcov[2] = {0}\n\tE_m = pcov[3] = {1}".format(*pcov))
           #print("Std.dev error is {}".format(np.sqrt(np.diag(pcov))))
           err = np.sqrt(np.diag(pcov))
           plt.savefig('_'.join([fit_type,p,name]))
           plt.close()
           return name, fit_type, popt, err

    except:
       print ('No linear extrapolate fit possible for {} {}'.format(p, name))
       plt.savefig('_'.join([fit_type,p,name]))
       plt.close()
       return None, None, [None,None], [None,None]


    return name, fit_type, popt, err


# user specifies which property, which code, which exchange to view
codes = 'VASP'
exchs = 'PBE'

# read in the Pade precision table with weight 4
# crossfilter down by specified code and exchange

#mydata = pd.read_csv('PadePrecs_final.csv')

mydata=pd.read_csv('main_with_precs_v2.csv')
code = mydata[mydata['code']==codes]
#print (code['exchange'])
exch_code = code[code['exchange']==exchs]

# variable definitions for fits and scatter data
dB_fits = { }

v0_fits = { }

B_fits = { }

E_vasp_pbe = exch_code[exch_code['property']=='E0']  ## VASP
dB_vasp_pbe = exch_code[exch_code['property']=='dB']
B_vasp_pbe = exch_code[exch_code['property']=='B']
v0_vasp_pbe = exch_code[exch_code['property']=='v0']

# separate data by element first and then by structure to rename
# the data labels as element_structure
for el in np.unique(E_vasp_pbe['element']):

   el_v0_vasp_pbe = v0_vasp_pbe[v0_vasp_pbe['element']==el]
   el_B_vasp_pbe = B_vasp_pbe[B_vasp_pbe['element']==el]
   el_dB_vasp_pbe = dB_vasp_pbe[dB_vasp_pbe['element']==el]
   el_E_vasp_pbe = E_vasp_pbe[E_vasp_pbe['element']==el]

   # split it down to structures by following
   # the unique structures found for energy data so that
   # element property is plotted against the same element
   # property
   for st in np.unique(el_E_vasp_pbe['structure']):

       st_el_v0_vasp_pbe = el_v0_vasp_pbe[el_v0_vasp_pbe['structure']==st]
       st_el_B_vasp_pbe = el_B_vasp_pbe[el_B_vasp_pbe['structure']==st]
       st_el_dB_vasp_pbe = el_dB_vasp_pbe[el_dB_vasp_pbe['structure']==st]
       st_el_E_vasp_pbe = el_E_vasp_pbe[el_E_vasp_pbe['structure']==st]


       if len(st_el_dB_vasp_pbe['perc_precisions'])==len(st_el_E_vasp_pbe['perc_precisions']):

           #count = count + 1
           name = '_'.join([st,el])
           # perform fit on data log(abs(perc_precision))
           print ('fitting dB for {}'.format(name))
           X = list(10*abs(st_el_E_vasp_pbe['perc_precisions']))
           Y = list(abs(st_el_dB_vasp_pbe['perc_precisions']))
           names = '_'.join(['R_dB',name])
           params_powR = regress_with_R(names,X,Y)
           print ('TESTING With R')
           print('{0} dB yields C= {1} +- {2} and M= {3} +- {4}'.format(*params_powR))
           #print ('Prediction curve {0}'.format(params_powR))
           dframe = pd.DataFrame({'X':X,'Y':Y})
           dframe.to_csv(name+'_dB_fits.csv', index=False)
           params_powR[5].to_csv(name+'_dB_preds.csv')

           # do a log-log fit for the power law
#           n, t, popt, err = evalFit(X,Y,name, fit_type='loglog', p='dB')
#           print('{0} {1} yields parameter a {2} +- {3} and parameter b {4} +- {5}'.format(n,t,popt[0],err[0],popt[1],err[1]))
           # do a log-lin fit for the exponential law
#           n, t, popt, err = evalFit(X,Y,name, fit_type='loglin', p='dB')
#           print('{0} {1} yields parameter a {2} +- {3} and parameter b {4} +- {5}'.format(n,t,popt[0],err[0],popt[1],err[1]))
           # do a lin-lin fit for the linear fit
#           n, t, popt, err = evalFit(X,Y,name, fit_type='linlin', p='dB')
#           print('{0} {1} yields parameter a {2} +- {3} and parameter b {4} +- {5}'.format(n,t,popt[0],err[0],popt[1],err[1]))

           dB_fits['_'.join([st,el])]=params_powR

           # test fits
           #frame= pd.DataFrame({'X': abs(10*st_el_E_vasp_pbe['perc_precisions']), 'Y': abs(st_el_dB_vasp_pbe['perc_precisions'])})
           #frame.to_csv('fit_linear_log.csv')
           # collect scatter pairs of

       if len(st_el_v0_vasp_pbe['perc_precisions'])==len(st_el_E_vasp_pbe['perc_precisions']):

           name = '_'.join([st,el])
           print ('fitting a0 for {}'.format(name))
           X = list(10*abs(st_el_E_vasp_pbe['perc_precisions']))
           Y = list(abs(st_el_v0_vasp_pbe['perc_precisions']))
           names = '_'.join(['R_v0',name])
           params_powR = regress_with_R(names, X,Y)
           print ('TESTING With R')
           print('{0} v0 yields C= {1} +- {2} and M= {3} +- {4}'.format(*params_powR))

           dframe = pd.DataFrame({'X':X,'Y':Y})
           dframe.to_csv(name+'_v0_fits.csv', index=False)
           params_powR[5].to_csv(name+'_v0_preds.csv')

           # do a log-log fit for the power law
#           n, t, popt, err = evalFit(X,Y,name, fit_type='loglog', p='a0')
#           print('{0} {1} yields parameter a {2} +- {3} and parameter b {4} +- {5}'.format(n,t,popt[0],err[0],popt[1],err[1]))
           # do a log-lin fit for the exponential law
#           n, t, popt, err = evalFit(X,Y,name, fit_type='loglin', p='a0')
#           print('{0} {1} yields parameter a {2} +- {3} and parameter b {4} +- {5}'.format(n,t,popt[0],err[0],popt[1],err[1]))
           # do a lin-lin fit for the linear fit
#           n, t, popt, err = evalFit(X,Y,name, fit_type='linlin', p='a0')
#           print('{0} {1} yields parameter a {2} +- {3} and parameter b {4} +- {5}'.format(n,t,popt[0],err[0],popt[1],err[1]))

           v0_fits['_'.join([st,el])]=params_powR


       if len(st_el_B_vasp_pbe['perc_precisions'])==len(st_el_E_vasp_pbe['perc_precisions']):

           name = '_'.join([st,el])
           # perform fit on data log(abs(perc_precision))
           print ('fitting B for {}'.format(name))
           X = list(10*abs(st_el_E_vasp_pbe['perc_precisions']))
           Y = list(abs(st_el_B_vasp_pbe['perc_precisions']))
           names = '_'.join(['R_B',name])
           params_powR = regress_with_R(names,X,Y)
           print ('TESTING With R')
           print('{0} B yields C= {1} +- {2} and M= {3} +- {4}'.format(name, *params_powR))

           dframe = pd.DataFrame({'X':X,'Y':Y})
           dframe.to_csv(name+'_B_fits.csv', index=False)
           params_powR[5].to_csv(name+'_B_preds.csv')
           # do a log-log fit for the power law
#           n, t, popt, err = evalFit(X,Y,name, fit_type='loglog', p='B')
#           print('{0} {1} yields parameter a {2} +- {3} and parameter b {4} +- {5}'.format(n,t,popt[0],err[0],popt[1],err[1]))
           # do a log-lin fit for the exponential law
#           n, t, popt, err = evalFit(X,Y,name, fit_type='loglin', p='B')
#           print('{0} {1} yields parameter a {2} +- {3} and parameter b {4} +- {5}'.format(n,t,popt[0],err[0],popt[1],err[1]))
           # do a lin-lin fit for the linear fit
#           n, t, popt, err = evalFit(X,Y,name, fit_type='linlin', p='B')
#           print('{0} {1} yields parameter a {2} +- {3} and parameter b {4} +- {5}'.format(n,t,popt[0],err[0],popt[1],err[1]))

           B_fits['_'.join([st,el])]=params_powR


#print (dB_fits.keys())

#col = ['red','blue']

#print (dB_fits.keys())
#print (B_fits.keys())
#print (a0_fits.keys())

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
print (type(v0_slope_m), type(B_slope_m), type(dB_slope_m))
#n, bins, patches = plt.hist(np.array([v0_slope_m, B_slope_m, dB_slope_m]).transpose())
#corrected_plaws = plaws.drop([6,31,11,10])

B_plaws = plaws[abs(plaws['B_C'])<5.0]
dB_plaws = B_plaws[abs(B_plaws['dB_C'])<5.0]
corrected_plaws = dB_plaws[abs(dB_plaws['v0_C']<6.0)]

corrected_plaws = plaws

print (corrected_plaws)
corrected_plaws.to_csv('PowerLawPairs_test_all.csv')
n, bins, patches = plt.hist(np.array([corrected_plaws['v0_M'], corrected_plaws['B_M'], corrected_plaws['dB_M']]).transpose())

plt.setp(patches[0], color="black", label='$a_0$')
plt.setp(patches[1], color="blue", label='$B$')
plt.setp(patches[2], color="red", label="$B'$")

plt.title("Sensitivity of Numerical Precision")
print (max([max(corrected_plaws['v0_M']), max(corrected_plaws['B_M']), max(corrected_plaws['dB_M'])]))
print (min([min(corrected_plaws['v0_M']), min(corrected_plaws['B_M']), min(corrected_plaws['dB_M'])]))
plt.xlim(0.0, max([max(corrected_plaws['v0_M']), max(corrected_plaws['B_M']), max(corrected_plaws['dB_M'])]))
plt.xlabel("Value of Slope $M$ (% per meV/atom)")
plt.ylabel("Number of Elements")
#plt.legend()

plt.savefig('{0}_{1}_slopes.pdf'.format(codes, exchs))
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
plt.close()


n, bins, patches = plt.hist(np.array([corrected_plaws['v0_x'], corrected_plaws['B_x'], corrected_plaws['dB_x']]).transpose())

plt.setp(patches[0], color="black")
plt.setp(patches[1], color="blue")
plt.setp(patches[2], color="red")
plt.title("Numerical Precision at {} meV/atom".format(str(meV_intercept)))
minx = min([min(corrected_plaws['v0_x']), min(corrected_plaws['B_x']), min(corrected_plaws['dB_x'])])
maxx = max([max(corrected_plaws['v0_x']), max(corrected_plaws['B_x']), max(corrected_plaws['dB_x'])])
print(minx, maxx)
labels  = ['$10^{'+str(s)+'}$' for s in [-6+x for x in range(0,8)]]
#print (labels)
ax = plt.gca()
ax.set_xticklabels(labels)
#plt.xlim(int(minx),1.0)
plt.xlim(-8,2)
plt.xlabel("Value of $\sigma_{V_0}, \sigma_{B_0}, \sigma_{B'}$ (%)")
plt.ylabel("Number of Elements")

#plt.show()
plt.tight_layout()
print ('use saved {0}_{1}_intercepts_{2}.pdf'.format(codes, exchs, str(meV_intercept)))
plt.savefig('{0}_{1}_intercepts_{2}.pdf'.format(codes, exchs, str(meV_intercept)))
plt.close()


print('finished!')



#ax[1].xlim(0.0,1.0)

#plt.xlabel("Value of intercept c")
#plt.ylabel("Number of elements")

#plt.title("Sensitivity of Relative Numerical Precision")
#plt.xlim(0.0,1.0)
#plt.xlabel("Value of slope m")
#plt.ylabel("Number of elements")
#plt.show()
