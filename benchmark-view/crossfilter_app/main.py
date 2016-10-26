# main.py that controls the whole app
# to run: just run bokeh serve --show crossfilter_app in the benchmark-view repo

from random import random
import os

from bokeh.layouts import column
from bokeh.models import Button, Dropdown
from bokeh.palettes import RdYlBu3
from bokeh.plotting import figure, curdoc


#### CROSSFILTER PART ##### >>> Module load errors throwing up how to do a relative import ?
from crossview.crossfilter.models import CrossFilter

#### DATA INPUT FROM REST API ######
#from benchmark.loader import load

#### DATA INPUT STRAIGHT FROM PANDAS for test purposes ####
import pandas as pd

##### PLOTTING PART -- GLOBAL FIGURE CREATION ########
# create a plot and style its properties

## gloabl data interface to come from REST API
vasp_data = pd.read_csv('./Data/Data.csv')

p = figure(x_range=(0, 100), y_range=(0, 100), toolbar_location=None)
#p.border_fill_color = 'black'
#p.background_fill_color = 'black'
p.outline_line_color = None
p.grid.grid_line_color = None


#### FORMAT OF DATA SENT TO WIDGET #######

# add a text renderer to out plot (no data yet)
r = p.text(x=[], y=[], text=[], text_color=[], text_font_size="20pt",
           text_baseline="middle", text_align="center")

i = 0

ds = r.data_source

##### WIDGET RESPONSES IN THE FORM OF CALLBACKS ######

# create a callback that will add a number in a random location
def callback():
    global i

    # BEST PRACTICE --- update .data in one step with a new dict
    new_data = dict()
    new_data['x'] = ds.data['x'] + [random()*70 + 15]
    new_data['y'] = ds.data['y'] + [random()*70 + 15]
    new_data['text_color'] = ds.data['text_color'] + [RdYlBu3[i%3]]
    new_data['text'] = ds.data['text'] + [str(i)]
    ds.data = new_data

    i = i + 1


#### The make crossfilter callback

#### make data loading as easy as possible for now straight from
#### the benchmark data csv file not from the API with the decorators

#### TO DO after we see that the crossfilter and new bokeh play nicely
##########: integrate with API and uncomment the decorators and data loader
#@bokeh_app.route("/bokeh/benchmark/")
#@object_page("benchmark")

#### RENDERERS OF WIDGETS #####

def make_bokeh_crossfilter():
    """The root crossfilter controller"""
    # Loading the dft data head as a
    # pandas dataframe
    # data = load("francesca_data_head")
    # use a straight pandas dataframe for now instead and follow the
    # BEST PRACTICE described above basically clean up the data object on each callback.
    new_data = dict() # data that will be given back on the callback
    data = pd.read_csv('crossfilter_app/Data/Data.csv') # our data that will be replaced by the API
    app = CrossFilter.create(df=data)
    print (type(app))
    # dont know what Crossfilter class really returns in terms of data but for testnig purposes lets
    # return something that is compatible with the new_data dictionary return in the
    # vanilla example through the global object ds.data
    # for example the x - y coordinates on the plots correspond to mins on the data set in k-point and value fields
    new_data['x'] = ds.data['x'] + [min(data['k-point'])]
    new_data['y'] = ds.data['y'] + [min(data['value'])]
    # other stuff default as in vanilla callback()
    new_data['text_color'] = ds.data['text_color'] + [RdYlBu3[i%3]]
    new_data['text'] = ds.data['text'] + [str(i)]
    # for test purposes to see actually what coordinate is getting plotted
    # it is always going to be the same duh beccause only one min exist in the dataset
    # its at x = 6, y = -12 ,
    # SUCESS learnt how to create a custom callback !!! that loads a CSV file and does something with it
    print ("New data from crossfilter", new_data)
    # finally assign to ds.data
    ds.data = new_data



def make_wflow_crossfilter(tags={'element_widget':['Cu', 'Pd', 'Mo'], 'code_widget':['VASP'], 'ExchCorr':['PBE']}):
    """
    demo crossfilter based on pure pandas dataframes that serves a data processing
    workflow that selects inputs from widgets

    args:
     tags: dict of selections by upto 3 widgets

    returns:
     dictionary of crossfiltered dataframes that can further be processed down the workflow
    """

    ## Actual widget controlled inputs ##

    # elements = tags['element']
    # exchanges = tags['ExchCorr']
    # propys = tags['code_widget']

    ## Demo user inputs for testing selects everything in the test csv : max data load ##

    elements = np.unique(vasp_data['element'])
    exchanges = np.unique(vasp_data['exchange'])
    propys = ['B','dB','a0']


    # final dictionary of crossfiltered dataframes
    crossfilts = {}
    # crossfiltering part - playing the role of the "Crossfilter class in bokeh.models"

    for pr in propys:
      for el in elements:
         for ex in exchanges:
            # crossfilter down to exchange and element
            elems = vasp_data[vasp_data['element']==el]
            exchs = elems[elems['exchange']==ex]
            # separate into properties, energy, kpoints
            p = exchs[exchs['property']==pr]
            e = exchs[exchs['property']=='e0']

            ##### *** Accuracy calculation based on default standards *** #####
            # choose reference from dict
            ref_e = expt_ref_prb[el][pr]
            ref_w = wien_ref[el][pr]
            # calculate percent errors on property - ACCURACY CALCULATION based on default standards
            props = [v for v in p['value'] ]
            percs_wien = [ (v - ref_w) / ref_w * 100 for v in p['value']]
            percs_prb = [ (v - ref_e) / ref_e * 100 for v in p['value']]
            kpts = [ k for k in p['k-point']]
            kpts_atom = [ k**3 for k in p['k-point'] ]
            ##### *** Accuracy calculation based on default standards *** #####

            ##### *** Calculate prec_sigma of energy *** #####
            energy = [ v for v in e['value']]
            end= len(energy) - 1
            prec_sigma = [ v - energy[end] for v in energy]

            # make data frame of kpoints, energy, percent errors on property
            if kpts and energy and props:
                NAME = '_'.join([el,ex,pr])
                Rdata =\
                pd.DataFrame({'Kpoints_size':kpts, 'Kpoints_atom_density':kpts_atom, 'Energy':energy, 'Prec_Sigma':prec_sigma , pr:props, 'percent_error_wien':percs_wien, 'percent_error_expt':percs_prb  })
                crossfilts[NAME] = Rdata



def calculate_prec(cross_df, automate= False):
    """
    function that calculates the prec_inf using R
    and returns a fully contructed plottable dataframe

    Args:
     cross_df: pandas dataframe containing the data
     automate: bool, a To do feature to automatically calculate the best fit

    Returns:
     dataframe contining the R added precision values to be
     received most always by the plotting commander.
    """
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    import rpy2.robjects.numpy2ri
    import rpy2.rinterface as rin


    stats = importr('stats')
    base = importr('base')
    # activate R environemnt in python
    rpy2.robjects.numpy2ri.activate()
    pandas2ri.activate()
    # read in necessary elements of crossfilts dataframe to R dataframe
    df = pd.DataFrame({'x': cross_df['Kpoints_atom_density'],
                       'y': cross_df['Energy']})
    ro.globalenv['dataframe']=df

    ### *** R used to obtain the fit on the data to calculate prec_inf *** ###
    # perform regression  - bokeh widgets can be used here to provide the inputs to the nls regression

    # some python to R translation of object names via the pandas - R dataframes
    y = df['y']
    x = df['x']
    l = len(y) - 1  # needed because R indexes list from 1 to len(list)

    # ***WIDGET inputs*** # OR AUTOMATE
    # the slider  inputs on starting point or can be automated also
    l1 = 3
    l2 = 0
    fitover = rin.SexpVector(list(range(l1,l-l2)), rin.INTSXP)

    # numeric entry widget for 'b' is plausible for user to choose best starting guess
    start_guess = {'a': y[l], 'b': 5}
    start=pandas2ri.py2ri(pd.DataFrame(start_guess,index=start_guess))

    # drop down list selection of model
    model = 'y~a*x/(b+x)'

    # Minimize function with weights and selection
    m = \
    stats.nls(model, start = start, algorithm = "port", subset = fitover, weights = x^2, data=base.as_symbol('dataframe'))

    # Estimation of goodness of fit
    g = stats.cor(y[l1:l-l2],stats.predict(m))

    # Report summary of fit, values and error bars
    print( base.summary(m).rx2('coefficients') )

    # Extrapolation value is given by a
    a = stats.coef(m)[1]

    # Calculation of precision
    prec = abs(y-a)

    # test print outs of the data ? how to render onto html like Shiny if necesary ?

    print("We learn that the converged value is: {0} and best precision achieved in the measurement is {1}".format(a, min(abs(prec))))

    cross_df['Energy_Prec_Inf'] = prec

    # close the R environments
    rpy2.robjects.numpy2ri.deactivate()
    pandas2ri.deactivate()

    return (cross_df)

def make_widgets():
    """
    main module that will control the rendering of UI widgets

    """
    pass

    
#### WIDGET CREATIONS ####

# OLD VANILLA
# add a button widget and configure with the call back
# button_basic = Button(label="Press Me")
# button_basic.on_click(callback)

# create a button for crossfilter
button_crossfilter = Button(label="Make Crossfilter")
button_crossfilter.on_click(make_bokeh_crossfilter)

#create a button for crossfilter_workflwo
button_w_crossfilter = Button(label="Make Crossfilter Workflow")
button_w_crossfilter.on_click(make_wflow_crossfilter)

# put the button and plot in a layout and add to the document
curdoc().add_root(column(button_crossfilter, button_w_crossfilter, p))
