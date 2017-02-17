import matplotlib
matplotlib.rcParams.update({'font.size': 18,'legend.fontsize':16, 'lines.markersize': 14})

import matplotlib.pyplot as plt
import pandas as pd

import numpy as np 

codes = 'VASP'
exchs = 'PBE'


def myLinFunc(x,M,C):
  return M*x+C




#corrected_plaws = pd.read_csv('histogram_data_without_outliers.csv')
corrected_plaws = pd.read_csv('outliers_removed_including_dev.csv')

n, bins, patches = plt.hist(np.array([corrected_plaws['v0_M'], corrected_plaws['B_M'], corrected_plaws['dB_M']]).transpose())

plt.setp(patches[0], color="black", label='$v_0$')
plt.setp(patches[1], color="blue", label='$B$')
plt.setp(patches[2], color="red", label="$B'$")

plt.title("Sensitivity of Numerical Precision")
print (max([max(corrected_plaws['v0_M']), max(corrected_plaws['B_M']), max(corrected_plaws['dB_M'])]))
print (min([min(corrected_plaws['v0_M']), min(corrected_plaws['B_M']), min(corrected_plaws['dB_M'])]))
plt.xlim(0.0, max([max(corrected_plaws['v0_M']), max(corrected_plaws['B_M']), max(corrected_plaws['dB_M'])]))
plt.xlabel("Value of Slope $M$ (% per meV/atom)")
plt.ylabel("Number of Elements")
#plt.legend()

plt.savefig('devs_v4_{0}_{1}_slopes.pdf'.format(codes, exchs))
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
plt.savefig('devs_v4_{0}_{1}_intercepts.pdf'.format(codes, exchs))
plt.close()

meV = 0.01
E_intercept = np.log(meV)

v0_x = [myLinFunc(E_intercept,corrected_plaws['v0_M'][n],corrected_plaws['v0_C'][n]) for n in corrected_plaws['v0_C'].index]
B_x = [myLinFunc(E_intercept,corrected_plaws['B_M'][n],corrected_plaws['B_C'][n]) for n in corrected_plaws['B_C'].index]
dB_x = [myLinFunc(E_intercept,corrected_plaws['dB_M'][n],corrected_plaws['dB_C'][n]) for n in corrected_plaws['dB_C'].index]

#v0_dev = [corrected_plaws['v0_dev'][n]) for n in corrected_plaws['v0_C'].index]
#B_dev = []
#dB_dev = []

n, bins, patches = plt.hist(np.array([v0_x, B_x, dB_x]).transpose())

plt.setp(patches[0], color="black")
plt.setp(patches[1], color="blue")
plt.setp(patches[2], color="red")
plt.title("Numerical Precision at {} meV/atom".format(str(meV)))
#minx = min([min(corrected_plaws['v0_x']), min(corrected_plaws['B_x']), min(corrected_plaws['dB_x'])])
#maxx = max([max(corrected_plaws['v0_x']), max(corrected_plaws['B_x']), max(corrected_plaws['dB_x'])])
#print(minx, maxx)
labels  = ['$10^{'+str(s)+'}$' for s in [-6+x for x in range(0,8)]]
#print (labels)
ax = plt.gca()
plt.xlim(-6,2)
ax.set_xticklabels(labels)
#plt.xlim(int(minx),1.0)
#plt.xlim(-8,2)
plt.xlabel("Value of $\sigma_{V_0}, \sigma_{B_0}, \sigma_{B'}$ (%)")
plt.ylabel("Number of Elements")

#plt.show()
plt.tight_layout()
print ('use saved {0}_{1}_intercepts_{2}.pdf'.format(codes, exchs, str(meV)))
plt.savefig('devs_v4_{0}_{1}_intercepts_{2}.pdf'.format(codes, exchs, str(meV)))
plt.close()


n, bins, patches = plt.hist(np.array([np.log10(corrected_plaws['v0_dev']), np.log10(corrected_plaws['B_dev']), np.log10(corrected_plaws['dB_dev'])]).transpose())

print (corrected_plaws['v0_dev'], corrected_plaws['B_dev'], corrected_plaws['dB_dev'])
plt.setp(patches[0], color="black")
plt.setp(patches[1], color="blue")
plt.setp(patches[2], color="red")
plt.title("Mean Numerical Precision Deviations".format(str(meV)))
minx = min([min(corrected_plaws['v0_dev']), min(corrected_plaws['B_dev']), min(corrected_plaws['dB_dev'])])
maxx = max([max(corrected_plaws['v0_dev']), max(corrected_plaws['B_dev']), max(corrected_plaws['dB_dev'])])
#print(minx, maxx)
labels  = ['$10^{'+str(s)+'}$' for s in [-4+x for x in range(0,6)]]
#print (labels)
ax = plt.gca()
#plt.xscale('log')
plt.xlim(-4,2)
ax.set_xticklabels(labels)
#plt.xlim(int(minx),1.0)
#plt.xlim(-8,2)
plt.xlabel("Value of $\sigma_{V_0}, \sigma_{B_0}, \sigma_{B'}$ (%)")
plt.ylabel("Number of Elements")

#plt.show()
plt.tight_layout()
print ('use saved {0}_{1}_deviations.pdf'.format(codes, exchs))
plt.savefig('devs_v4_{0}_{1}_deviations.pdf'.format(codes, exchs))
plt.close()

print('finished!')

