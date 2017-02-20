from os.path import dirname, join

import pandas as pd
import numpy as np
from collections import OrderedDict

from bokeh.layouts import row, widgetbox, column
from bokeh.models import Select, Div, Column, HoverTool, ColumnDataSource, PanTool, ResetTool, BoxZoomTool
from bokeh.palettes import Spectral5
from bokeh.plotting import curdoc, figure
from bokeh.sampledata.periodic_table import elements

##### loading the data which the API needs to take care fo ###### 
#from bokeh.sampledata.autompg import autompg

#df = autompg.copy()
#print (df.columns)
SIZES = list(range(6, 22, 3))
COLORS = Spectral5




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
#df_ref = pd.read_json('./Data/Ref.json')

#df_obs.percent_accuracy = [ ( (x['value'] - df_ref[x['element']][x['property']]) / df_ref[x['element']][x['property']] ) * 100 
#                           for x in df_obs.iterrows() ]

# df_pade_prec
#df_obs.percent_pade_prec = [ ( (x['value'] - df_pade_prec[x['element']][x['property'] ] ) / df_pade_prec[x['element']][x['property']])  * 100 for x in df_obs.iterrows()]

# df_sigma_prec
#df_obs.percent_sigma_prec = [ ( (x['value'] - df_sigma_prec[x['element']][x['property'] ] ) / df_sigma_prec[x['element']][x['property']])  * 100 for x in df_obs.iterrows()]

# dividing data into gneeral discretes, continuous and quantileables

columns = sorted(df_obs.columns) #+ sorted(df.columns)
#print columns
print ([(x, df_obs[x].dtype) for x in columns])
#print ([(x,df[x].dtype) for x in sorted(df.columns)])
discrete = [x for x in columns if df_obs[x].dtype == object] #+ [x for x in columns if df_obs[x].dtype == object]
print (discrete)
continuous = [x for x in columns if x not in discrete]
#print continuous
#quantileable = [x for x in continuous if len(df[x].unique()) > 20]
#print quantileable
####################################################################

##divide data into plottables and non plottables (aggregates or 3D plottables) , 
#keep it to 2D plottables for now, this is known from the column names themselves

plottables =  ['k-point', 'value', 'smearing']

non_plottables = [ x for x in columns if x not in plottables ] # for aggregates

_elements = list(np.unique(df_obs['element']))
exchanges = list(np.unique(df_obs['exchange']))
properties = list(np.unique(df_obs['property']))
codes = list(np.unique(df_obs['code']))

# which sets of k-point and value to string together ? any unit transformations on the dataset values or k-point 

## have another dataframe (mongo collection) for the reference standards to compute the accuracy (uniquely identified by the element SAME standard should apply to all codes/exchanges/elements.   

def create_figure():


    # original autpmpg test 

    xs =df_obs[x.value].values
#    print (type(sorted(set(xs))))
    # read the data from the df
#    xs = df_obs[x.value].values
    ys = df_obs[y.value].values
    x_title = x.value.title()
    y_title = y.value.title()

    df_select2d = pd.DataFrame({x.value:xs, y.value:ys})
#    print (df_select2d)
    kw = dict()
#    if x.value in continuous:
#        kw['x_range'] = sorted(set(xs))
#    print (type(kw['x_range']))
#    if y.value in continuous:
#        kw['y_range'] = sorted(set(ys))
#    print (type(kw['y_range']))
    kw['title'] = "%s vs %s" % (x_title, y_title)

    source = ColumnDataSource(ColumnDataSource.from_df(df_select2d))
#    print (source.columns)
    hover = HoverTool()
    pan = PanTool()
    bzoom = BoxZoomTool()
#    hover.tooltips = OrderedDict([('k-point', '$k-point'),('value', '$value')])
#    hover = HoverTool(tooltips=[("(x, $x)", "(y, $y)")])

    p = figure(plot_height=600, plot_width=800, tools=[hover, pan, bzoom], **kw)
    p.xaxis.axis_label = x_title

    p.yaxis.axis_label = y_title

    hover = p.select(dict(type=HoverTool))
    hover.tooltips = OrderedDict([('k-point', '$k-point'),('value', '$value')])

    # sets the xaxis
    if x.value in continuous:
        p.xaxis.major_label_orientation = pd.np.pi / 4


    sz = 9
    if size.value != 'None':
        groups = pd.qcut(df[size.value].values, len(SIZES))
        sz = [SIZES[xx] for xx in groups.codes]

    c = "#31AADE"
    if color.value != 'None':
        groups = pd.qcut(df[color.value].values, len(COLORS))
        c = [COLORS[xx] for xx in groups.codes]
    p.circle(x='k-point', y='value', source=source)#, color=c, size=sz, line_color="white", alpha=1.0, hover_color='blue')#, hover_alpha=1.0)

    return p


def update(attr, old, new):
    layout.children[1] = create_figure()


############## Header Content from description.html  #################
content_filename = join(dirname(__file__), "description.html")

description = Div(text=open(content_filename).read(),
                  render_as_text=False, width=600)





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



######### CREATES CROSSFILTER ##########################


# decide if all columns or crossfilter down to sub properties

# The crossfilter widgets

# first select code this crossfilters the available options to 
# available exchanges and elements 
code = Select(title='Code', value='vasp', options=codes)
code.on_change('value', update)


# second select exchange 

element = Select(title='Element', value='Cu', options=_elements)
element.on_change('value', update)

exchange = Select()


# The plotter widgets 

x = Select(title='X-Axis', value='k-point', options=plottables)
x.on_change('value', update)

y = Select(title='Y-Axis', value='value', options=plottables)  
y.on_change('value', update)

z = Select(title='Z-Axis', value='None', options=plottables)
z.on_change('value', update)



size = Select(title='Size', value='None', options=['None'] )
size.on_change('value', update)

color = Select(title='Color', value='None', options=['None'] )
color.on_change('value', update)

controls = widgetbox([x, y, color, size], width=200)
layout = column(description, ptable, controls, create_figure())

curdoc().add_root(layout)
curdoc().title = "DFT Benchmark"
