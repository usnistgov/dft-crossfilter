from os.path import dirname, join

import pandas as pd

from bokeh.layouts import row, widgetbox, column
from bokeh.models import Select, Div, Column, HoverTool, ColumnDataSource
from bokeh.palettes import Spectral5
from bokeh.plotting import curdoc, figure
from bokeh.sampledata.periodic_table import elements

##### loading the data which the API needs to take care fo ###### 
#from bokeh.sampledata.autompg import autompg

#df = autompg.copy()

SIZES = list(range(6, 22, 3))
COLORS = Spectral5


# data cleanup
#df.cyl = [str(x) for x in df.cyl]
#df.origin = [ORIGINS[x-1] for x in df.origin]

#df['year'] = [str(x) for x in df.yr]
#del df['yr']

#df['mfr'] = [x.split()[0] for x in df.name]
#df.loc[df.mfr=='chevy', 'mfr'] = 'chevrolet'
#df.loc[df.mfr=='chevroelt', 'mfr'] = 'chevrolet'
#df.loc[df.mfr=='maxda', 'mfr'] = 'mazda'
#df.loc[df.mfr=='mercedes-benz', 'mfr'] = 'mercedes'
#df.loc[df.mfr=='toyouta', 'mfr'] = 'toyota'
#df.loc[df.mfr=='vokswagen', 'mfr'] = 'volkswagen'
#df.loc[df.mfr=='vw', 'mfr'] = 'volkswagen'
#del df['name']

df_obs = pd.read_csv('./Data/Data.csv')

# single reference standard this can be an on request 
# basis input as well 
df_ref = pd.read_json('./Data/Ref.json')

# dividing data into gneeral discretes, continuous and quantileables

columns = sorted(df_obs.columns)
print columns
#discrete = [x for x in columns if df[x].dtype == object]
#print discrete
#continuous = [x for x in columns if x not in discrete]
#print continuous
#quantileable = [x for x in continuous if len(df[x].unique()) > 20]
#print quantileable
####################################################################

##divide data into plottables and non plottables (aggregates or 3D plottables) , keep it to 2D plottables for now, this is known from the column names themselves

plottables =  ['k-point', 'value', 'smearing']

non_plottables = [ x for x in columns if x not in plottables ] # for aggregates

elements = np.unique(df_obs['element'])
exchanges = np.unique(df_obs['exchange'])
properties = np.unique(df_obs['property'])
codes = np.unique(df_obs['code'])

# which sets of k-point and value to string together ? any unit transformations on the dataset values or k-point 

## have another dataframe (mongo collection) for the reference standards to compute the accuracy (uniquely identified by the element SAME standard should apply to all codes/exchanges/elements.   

def create_figure():

    # read the data from the df
    xs = df_obs[x.value].values
    ys = df_obs[y.value].values
    x_title = x.value.title()
    y_title = y.value.title()

    # set autoscale, title and x-y labels of the plot 
    kw = dict()
#    if x.value in discrete:
#    kw['x_range'] = sorted(set(xs))
#    if y.value in discrete:
#    kw['y_range'] = sorted(set(ys))

    kw['title'] = "%s vs %s" % (x_title, y_title)

    p = figure(plot_height=600, plot_width=800, tools='pan,box_zoom,reset,hover', **kw)
    p.xaxis.axis_label = x_title

    p.yaxis.axis_label = y_title


    # sets the xaxis
#    if x.value in discrete:
    p.xaxis.major_label_orientation = pd.np.pi / 4

    ## sizes and colors of the points for now turn them off 

#    sz = 9
#    if size.value != 'None':
#        groups = pd.qcut(df[size.value].values, len(SIZES))
#        sz = [SIZES[xx] for xx in groups.codes]

#    c = "#31AADE"
#    if color.value != 'None':
#        groups = pd.qcut(df[color.value].values, len(COLORS))
#        c = [COLORS[xx] for xx in groups.codes]
#    p.circle(x=xs, y=ys, color=c, size=sz, line_color="white", alpha=0.6, hover_color='white', hover_alpha=0.5)

     # scatter plot of circles of the chosen values 
     p.circle(x=xs, y=ys, alpha=0.6, hover_color='white', hover_alpha=0.5)

    return p


def update(attr, old, new):
    layout.children[1] = create_figure()




def update_crossfilter(tag):
    ## options for the tag are: code, exchange, element, property
    if tag == code: 
       # code selection redefines the element and exchange selection basically, shoudl include properties also but thta's uniform so ok for now
       code_selected = code.value
       df_obs = df_obs[df_obs['code']==code_selected] #by default crossfilters to all vasp code

       exchanges = np.unique(df_obs['exchange'])
       elements = np.unique(df_obs['element'])

       element = Select(title='Element', value = elements[0], options=elements ) #### here is where the periodic table widget can be updated instead as well
#       element.on_change('value', update_crossfilter(tag=element ))
       exchange = Select(title='Exchange', value = exchanges[0], options=exchanges)
#       exchange.on_change('value', update_crossfilter(tag=exchange))

    elif tag == exchange:
       # exchnage selection redefines the code and element selection basically
       exchange_selected = exchange.value
       df_obs = df_obs[df_obs['exchange']==exchange_selected]

       codes = np.unique(df_obs['code'])
       elements = np.unique(df_obs['element'])

       code = Select(title='Code', value = codes[0], options=codes)
#       code.on_change('value', update_crossfilter(tag=code))
       element = Select(title='Element', value = elements[0], options=elements ) #### here is where the periodic table widget can be updated instead as well 
#       element.on_change('value', update_crossfilter(tag=element) )

    elif tag == element:
       # element selection redefines the exchange and code selection
       element_selected = element.value # can in principle be a list also from a multi-select ? 
       df_obs = df_obs[df_obs['element']==element_selected]

       exchanges = np.unique(df_obs['exchange'])
       exchange = Select(title='Exchange', value = exchanges[0], options=exchanges)

       codes = np.unique(df_obs['code'])
       code = Select(title='Code', value = codes[0], options=codes)

       properties = np.unique(df_obs['property']) # redundant because this will mostly not change 
       prop = Select(title='Property', value = 'B', options=properties ) ####once the property is selected correctly 

    elif tag == decide:
       decision = decide.value
       if decision == 'Yes':
           is_plot=True      # break out of the recursion crossfilter selection
       else:
           is_plot=False

### can have an optional refresh button to reset the crossfilter. 

############## Header Content from description.html  #################
content_filename = join(dirname(__file__), "description.html")

description = Div(text=open(content_filename).read(),
                  render_as_text=False, width=600)



#### PERIODIC TABLE plot ############################### 

romans = ["I", "II", "III", "IV", "V", "VI", "VII"]

elements["atomic mass"] = elements["atomic mass"].astype(str)

elements["period"] = [romans[x-1] for x in elements.period]
elements = elements[elements.group != "-"]

group_range = [str(x) for x in range(1, 19)]

colormap = {
    "alkali metal"         : "#a6cee3",
    "alkaline earth metal" : "#1f78b4",
    "halogen"              : "#fdbf6f",
    "metal"                : "#b2df8a",
    "metalloid"            : "#33a02c",
    "noble gas"            : "#bbbb88",
    "nonmetal"             : "#baa2a6",
    "transition metal"     : "#e08e79",
}

source = ColumnDataSource(
    data=dict(
        group=[str(x) for x in elements["group"]],
        period=[str(y) for y in elements["period"]],
        symx=[str(x)+":0.1" for x in elements["group"]],
        numbery=[str(x)+":0.8" for x in elements["period"]],
        massy=[str(x)+":0.15" for x in elements["period"]],
        namey=[str(x)+":0.3" for x in elements["period"]],
        sym=elements["symbol"],
        name=elements["name"],
        cpk=elements["CPK"],
        atomic_number=elements["atomic number"],
        electronic=elements["electronic configuration"],
        mass=elements["atomic mass"],
        type=elements["metal"],
        type_color=[colormap[x] for x in elements["metal"]],
    )
)

ptable = figure(title="Periodic Table", tools="hover,save",
           x_range=group_range, y_range=list(reversed(romans)))
ptable.plot_width = 1200
ptable.toolbar_location = None
ptable.outline_line_color = None

ptable.rect("group", "period", 0.9, 0.9, source=source,
       fill_alpha=0.6, color="type_color")

text_props = {
    "source": source,
    "angle": 0,
    "color": "black",
    "text_align": "left",
    "text_baseline": "middle"
}

ptable.text(x="symx", y="period", text="sym",
       text_font_style="bold", text_font_size="15pt", **text_props)

ptable.text(x="symx", y="numbery", text="atomic_number",
       text_font_size="9pt", **text_props)

ptable.text(x="symx", y="namey", text="name",
       text_font_size="6pt", **text_props)

ptable.text(x="symx", y="massy", text="mass",
       text_font_size="5pt", **text_props)

ptable.grid.grid_line_color = None

ptable.select_one(HoverTool).tooltips = [
    ("name", "@name"),
    ("atomic number", "@atomic_number"),
    ("type", "@type"),
    ("atomic mass", "@mass"),
    ("electronic configuration", "@electronic"),
]


is_plot = True


######### CREATES CROSSFILTER ##########################


# decides what are the plottables

# The crossfilter widgets -- add on feature - multiselection of data 


# code based crossfilter 
code = Select(title='Code', value='vasp', options=codes)
code.on_change('value', update_crossfilter(tag=code))


# exchange based crossfilter 
exchange = Select(title='exchange', value='PBE', options=exchanges
exchange.on_change('value', update_crossfilter(tag=exchange))

# element based crossfilter -- can be periodic table widget instead
element = Select(title='Element', value='Cu', options=elements)
element.on_change('value', update_crossfilter(tag=element))

# property based crossfilter
prop = Select(title='Property', value='E0', options=properties)
prop.on_change('value', update_crossfilter(tag=prop))


# finish the crossfilter selection and go to plot -- can be a button widget also 
decide = Select(title='Plot?', value='Yes', options =['Yes', 'No'])
decide.on_change('value', update_crossfilter(tag=decide))

if is_plot:
     # The plotter widgets 

     ## plot if is_plot is ready from the crossfiltering ## 

     x = Select(title='X-Axis', value='k-point', options=plottables)
     x.on_change('value', update)

     y = Select(title='Y-Axis', value='value', options=plottables)  
     y.on_change('value', update)

#     z = Select(title='Z-Axis', value='None', options=plottables)
#     z.on_change('value', update)



     size = Select(title='Size', value='None', options=['None'] + quantileable)
     size.on_change('value', update)

     color = Select(title='Color', value='None', options=['None'] + quantileable)
     color.on_change('value', update)

# final layout widgets first
controls = widgetbox([code, exchange, element, prop, x, y, color, size], width=200)
layout = column(description, ptable, controls, create_figure())

curdoc().add_root(layout)
curdoc().title = "DFT Benchmark"
