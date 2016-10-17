# main.py that controls the whole app
# to run: just run bokeh serve --show crossfilter_app in the benchmark-view repo

from random import random
import os

from bokeh.layouts import column
from bokeh.models import Button, Dropdown
from bokeh.palettes import RdYlBu3
from bokeh.plotting import figure, curdoc


#### CROSSFILTER PART ##### >>> Module load errors throwing up how to do a relative import ?
# from .crossview.crossfilter.models import CrossFilter

#### DATA INPUT FROM REST API ######
#from benchmark.loader import load

#### DATA INPUT STRAIGHT FROM PANDAS for test purposes ####
import pandas as pd

##### PLOTTING PART -- GLOBAL FIGURE CREATION ########
# create a plot and style its properties

p = figure(x_range=(0, 100), y_range=(0, 100), toolbar_location=None)
#p.border_fill_color = 'black'
#p.background_fill_color = 'black'
p.outline_line_color = None
p.grid.grid_line_color = None

# add a text renderer to out plot (no data yet)
r = p.text(x=[], y=[], text=[], text_color=[], text_font_size="20pt",
           text_baseline="middle", text_align="center")

i = 0

ds = r.data_source

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

def make_crossfilter():
    """The root crossfilter controller"""
    # Loading the dft data head as a
    # pandas dataframe
    # data = load("francesca_data_head")
    # use a straight pandas dataframe for now instead and follow the
    # BEST PRACTICE described above basically clean up the data object on each callback.
    new_data = dict() # data that will be given back on the callback
    data = pd.read_csv('crossfilter_app/Data/Data.csv') # our data
    # app = CrossFilter.create(df=autompg)
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

#### WIDGET CREATIONS ####

# add a button widget and configure with the call back
button_basic = Button(label="Press Me")
button_basic.on_click(callback)

# create a button for crossfilter
button_crossfilter = Button(label="Make Crossfilter")
button_crossfilter.on_click(make_crossfilter)
# put the button and plot in a layout and add to the document
curdoc().add_root(column(button_basic, button_crossfilter, p))
