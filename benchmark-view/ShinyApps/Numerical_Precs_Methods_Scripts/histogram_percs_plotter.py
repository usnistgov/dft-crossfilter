import matplotlib
matplotlib.rcParams.update({'font.size': 18,'legend.fontsize':16, 'lines.markersize': 18})
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

codes = 'VASP'
exchs = 'LDA'

corrected_plaws = pd.read_csv('PowerLawPairs_test_LDA.csv')#plaws

total_elements = len(corrected_plaws['v0_names'])
#print (corrected_plaws)

#corrected_plaws.to_csv('PowerLawPairs_test_v6.csv')
from matplotlib.ticker import FuncFormatter


def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(100 * y)

    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'

# Create the formatter using the function to_percent. This multiplies all the
# default labels by 100, making them all percentages
formatter = FuncFormatter(to_percent)

my_hist_array = np.array([corrected_plaws['v0_M'], corrected_plaws['B_M'], corrected_plaws['dB_M']]).transpose()

my_weights = np.ones_like(my_hist_array)/float(len(my_hist_array))

n, bins, patches = plt.hist(my_hist_array, weights=my_weights)
#print(patches[0], n, bins)
plt.setp(patches[0], color="black", label='$a_0$')
plt.setp(patches[1], color="blue", label='$B$')
plt.setp(patches[2], color="red", label="$B'$")

plt.title("Sensitivity of Numerical Precision")
print (max([max(corrected_plaws['v0_M']), max(corrected_plaws['B_M']), max(corrected_plaws['dB_M'])]))
print (min([min(corrected_plaws['v0_M']), min(corrected_plaws['B_M']), min(corrected_plaws['dB_M'])]))
plt.xlim(0.0, max([max(corrected_plaws['v0_M']), max(corrected_plaws['B_M']), max(corrected_plaws['dB_M'])]))
plt.xlabel("Value of Slope $M$ (% per meV/atom)")
plt.ylabel("% of Elements")
#plt.legend()
plt.gca().yaxis.set_major_formatter(formatter)
plt.tight_layout()
#ax = plt.gca()
#print ('y limuts for M', ax.get_ylim())
plt.savefig('stables_{0}_{1}_slopes.pdf'.format(codes, exchs))
plt.savefig('PPt_stables_LDA_{0}_{1}_slopes.png'.format(codes, exchs))
plt.close()
## Intercepts
#fig,ax = plt.subplots()

my_hist_array = np.array([corrected_plaws['v0_C'], corrected_plaws['B_C'], corrected_plaws['dB_C']]).transpose()
weights = np.ones_like(my_hist_array)/float(len(my_hist_array))

n, bins, patches = plt.hist(my_hist_array, weights=weights)

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
plt.ylabel("% of Elements")

#plt.show()
plt.gca().yaxis.set_major_formatter(formatter)
plt.tight_layout()
plt.savefig('stables_{0}_{1}_intercepts.pdf'.format(codes, exchs))
plt.savefig('PPt_stables_LDA_{0}_{1}_intercepts.png'.format(codes, exchs))
plt.close()


my_hist_array = np.array([corrected_plaws['v0_x'], corrected_plaws['B_x'], corrected_plaws['dB_x']]).transpose()
weights = np.ones_like(my_hist_array)/float(len(my_hist_array))

n, bins, patches = plt.hist(my_hist_array, weights=weights)

plt.setp(patches[0], color="black")
plt.setp(patches[1], color="blue")
plt.setp(patches[2], color="red")
plt.title("Numerical Precision at {} meV/atom".format(str(meV_intercept)))
minx = min([min(corrected_plaws['v0_x']), min(corrected_plaws['B_x']), min(corrected_plaws['dB_x'])])
maxx = max([max(corrected_plaws['v0_x']), max(corrected_plaws['B_x']), max(corrected_plaws['dB_x'])])
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
plt.gca().yaxis.set_major_formatter(formatter)
plt.tight_layout()
print ('use saved v6_stables_{0}_{1}_intercepts_{2}.pdf'.format(codes, exchs, str(meV_intercept)))
plt.savefig('LDA_stables_{0}_{1}_intercepts_{2}.pdf'.format(codes, exchs, str(meV_intercept)))
plt.savefig('PPt_stables_LDA_{0}_{1}_intercepts_{2}.png'.format(codes, exchs, str(meV_intercept)))
plt.close()


print('finished!')
