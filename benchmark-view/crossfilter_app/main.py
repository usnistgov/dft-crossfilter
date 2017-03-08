import os
from os.path import dirname, join

import pandas as pd
import numpy as np

from bokeh.layouts import row, widgetbox, column
from bokeh.models import Select, Div, Column, HoverTool, ColumnDataSource, Button
from bokeh.palettes import Spectral5
from bokeh.plotting import curdoc, figure
from bokeh.sampledata.periodic_table import elements


df_obs = pd.read_csv('./crossfilter_app/Data/Data.csv')
print ('read df_obs')
# single reference standard this can be an on request
# basis input as well
#df_ref = pd.read_json('./Data/Ref.json')

# dividing data into gneeral discretes, continuous and quantileables

columns = sorted(df_obs.columns) #+ sorted(df.columns)
#print columns
#print ('Here', [(x, df_obs[x].dtype) for x in columns])
#print ([(x,df[x].dtype) for x in sorted(df.columns)])
discrete = [x for x in columns if df_obs[x].dtype == object] #+ [x for x in columns if df_obs[x].dtype == object]
#print ('and here', discrete)
continuous = [x for x in columns if x not in discrete]
#print continuous
#quantileable = [x for x in continuous if len(df[x].unique()) > 20]
#print quantileable
####################################################################

##divide data into plottables and non plottables (aggregates or 3D plottables) ,
#keep it to 2D plottables for now, this is known from the column names themselves

plottables =  ['k-point', 'value', 'perc_precisions']

non_plottables = [ x for x in columns if x not in plottables ] # for aggregates

structures = list(np.unique(df_obs['structure']))
_elements = list(np.unique(df_obs['element']))
print (_elements)
exchanges = list(np.unique(df_obs['exchange']))
properties = list(np.unique(df_obs['property']))
codes = list(np.unique(df_obs['code']))

# which sets of k-point and value to string together ? any unit transformations on the dataset values or k-point

## have another dataframe (mongo collection) for the reference standards to compute the accuracy (uniquely identified by the element SAME standard should apply to all codes/exchanges/elements.

############## Header Content from description.html  #################

content_filename = join(dirname(__file__), "description.html")

description = Div(text=open(content_filename).read(),
                  render_as_text=True, width=600)


# periodic table widget
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
# plot the periodic layout
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
#source_data = pd.DataFrame({})#ColumnDataSource(data=dict())

class CrossFiltDFs():

    def __init__(self,struct_df=None,elem_df=None,prop_df=None,\
                 plot_data=None, code_df=None, exchange_df=None):
        self.struct_df = struct_df
        self.elem_df = elem_df
        self.prop_df = prop_df
        self.code_df = code_df
        self.exchange_df = exchange_df
        self.plot_data = plot_data

    def crossfilter_by_tag(self,df, tag):
        """
        a crossfilter that can recursivly update the unique options
        in the UI based on prioir selections

        returns crossfiltered by tag crossfilter {'element': 'Ag'}
        """
        col,spec= list(tag.items())[0]
        #col, spec = tag.items()
        return df[df[col]==spec]



    def crossfilter_by_choices(struct_choice, elem_choice, prop_choice, code_choice,exchange_choice):
        """
        crossfilter control based on UI choices.
        crossfilters and returns the dataframe after crossfiltering

        may be deprecated
        """
        # first selects the element
        struct_df  = df_obs[df_obs['structure']==struct_choice]
        #print (struct_df)
        print ('elem',elem_choice)
        elem_struct_df = struct_df[struct_df['element']==elem_choice]
        print (elem_struct_df)
        prop_elem_struct_df = elem_struct_df[elem_struct_df['property']==prop_choice]
        print(prop_elem_struct_df)
        code_prop_elem_struct_df = \
           prop_elem_struct_df[prop_elem_struct_df['code']==code_choice]
        print (code_prop_elem_struct_df)
        exchange_code_prop_elem_struct_df =\
        code_prop_elem_struct_df[code_prop_elem_struct_df['exchange']==exchange_choice]
        print ('this is the crossfilt being returned', exchange_code_prop_elem_struct_df)
        return exchange_code_prop_elem_struct_df


    def create_figure(self,dataset,datplot='Init',plot_type=None):
        """
        figure and plot creation for a given dataset
        TODO: enable support for multiple selection
        refactor to a simple figure creator and
        add helper functions for the plots
        """
        kw = dict()

        print (dataset)
        if datplot=='Init':
           # if data is to be plotted
           xs =dataset[x.value].values
           ys = dataset[y.value].values
           self.xs_init = xs
           self.ys_init = ys
           x_title = x.value.title()
           y_title = y.value.title()

           kw['title'] = "%s vs %s" % (y_title, x_title)

           if x.value=='k-point':
              kw['x_axis_type'] = 'log'

           elif x.value == 'perc_precisions' and y.value == 'perc_precisions':
              kw['x_axis_type'] = 'log'
              kw['y_axis_type'] = 'log'

           p = figure(plot_height=600, plot_width=800, tools='pan,wheel_zoom,reset,hover', **kw)
           p.xaxis.axis_label = x_title

           p.yaxis.axis_label = y_title


    # sets the xaxis
           if x.value in continuous:
              p.xaxis.major_label_orientation = pd.np.pi / 4

          #if x.value == 'k-point':
        #    xs = [k**3 for k in xs]
    #sz = 9
    #if size.value != 'None':
    #    groups = pd.qcut(df[size.value].values, len(SIZES))
    #    sz = [SIZES[xx] for xx in groups.codes]

    #c = "#31AADE"
    #if color.value != 'None':
    #    groups = pd.qcut(df[color.value].values, len(COLORS))
    #    c = [COLORS[xx] for xx in groups.codes]
           p.scatter(x=xs, y=ys, line_color="white", alpha=1.0, hover_color='blue', hover_alpha=1.0)
           self.p = p
           return self.p

        elif datplot == 'Add':
           # add a plot to figure, from statistical analysis
           if plot_type == 'plot_pade':

               pade_order = self.analysis_results['Order']
               pade_extrapolate = self.analysis_results['Extrapolate']

               # create precisions based on the extrapolate

               xs = self.xs_init
               ys = [abs(y-pade_extrapolate) for y in self.ys_init]

               # print (xs,ys,len(xs),len(ys))

               self.p.line(x=xs, y=ys, line_color="red", alpha=1.0, hover_color='blue', hover_alpha=1.0)
               return self.p

        else:
          # clear the figure by plotting an empty figure
          xs = []
          ys = []
          p = figure(plot_height=600, plot_width=800, tools='pan,wheel_zoom,reset,hover', **kw)
          p.scatter(x=xs, y=ys, line_color="white", alpha=1.0, hover_color='blue', hover_alpha=1.0)
          return p

# The crossfilter widgets
    def update(self, attr, old, new):
       print ('Attribute', attr, 'OLD', old, 'NEW', new)
       #print (len(layout.children))
       print ('executes here on update')#, exchange_df)

    def update_code(self):
        """
        update for the code selection
        """
        self.code_df = self.elem_df[self.elem_df['code'] == code.value].dropna()

    def update_exchange(self):
        """
        update the exchange
        """
        self.exchange_df = self.code_df[self.code_df['exchange']== exchange.value].dropna()

    def update_element(self):
        print ('Updating element down selection for property')
        self.elem_df = df_obs[df_obs['element'] == element.value].dropna()
        self.plot_data = self.elem_df

    def update_struct(self):
       #print ('Updating struct down selection for element')
       #print ("struct.value",struct.value)
       self.struct_df = self.exchange_df[self.exchange_df['structure'] == struct.value].dropna()
       self.plot_data = self.struct_df
       print ('finished callback to update layout')


    def update_prop(self):
       #print ('Updating struct down selection for element')
       #print (prop.value)
       self.prop_df = self.struct_df[self.struct_df['property'] == prop.value].dropna()
       #print ('The final dict', self.prop_df.to_dict(orient='list'))
       self.plot_data = self.prop_df

    def update_x(self):
        pass

    def update_y(self):
        pass

    def update_crossfilter(self):
       print ('Triggering crossfilter')
       print (type(self.plot_data))
       print (np.unique(self.plot_data['property']))
       layout.children[3] = self.create_figure(self.plot_data)

    def clear_crossfilter(self):
        """
        clear the figure and crossfilter
        """
        print ('Trigger clear')
        self.struct_df = None
        self.elem_df = None
        self.prop_df = None
        self.code_df = None
        self.exchange_df = None
        self.plot_data = None
        layout.children[3] = self.create_figure(self.plot_data)

    def analysis_callback(self):
        """
        calls the Pade analysis on the current plot data
        TODO:
        NOTE: check if this is a data set that is a single scatter
        FEATUREs that could be added: plot the Pade for multiple selections
        """
        print ('called Pade analysis')
        # writes out the crossfiltered plot data on the server
        crossfilt = self.plot_data[['k-point','value']]
        crossfilt.columns=['Kpt','P']
        crossfilt.to_csv('crossfilter_app/Rdata.csv')
        os.system('Rscript crossfilter_app/non_err_weighted_nls.R')
        self.analysis_results = pd.read_csv('crossfilter_app/Result.csv')
        print ('executed R script on crossfiltered data')
        layout.children[3] = self.create_figure(self.plot_data, datplot='Add', plot_type='plot_pade')

#def update():
#    pass
    #source_data = CF.plot_data

# initialize the crossfilter instance
CF = CrossFiltDFs()

# define the selection widgets for code, exchange,
# TODO: enable widgets that support multi-selection
# Elements selection widget from a periodic table

code = Select(title='Code', value=codes[0], options=codes)
code.on_change('value', lambda attr, old, new: CF.update_code())

exchange = Select(title='ExchangeCorrelation', value=exchanges[0], options=exchanges)
exchange.on_change('value', lambda attr, old, new: CF.update_exchange())

struct = Select(title='Structure', value=structures[0], options=structures)
struct.on_change('value', lambda attr, old, new: CF.update_struct())

#elem_options = list(np.unique(struct_df['element']))
element = Select(title='Element', value=_elements[0], options=_elements)
element.on_change('value', lambda attr, old, new: CF.update_element())


#prop_options = list(np.unique(elem_df['property']))
prop = Select(title='Property', value=properties[0], options=properties)
prop.on_change('value', lambda attr, old, new: CF.update_prop())


#code_options = list(np.unique(prop_df['code']))
#code = Select(title='Code', value=code_options[0], options=code_options)
#code.on_change('value', update_code)
#code_df = crossfilter_by_tag(prop_df, {'property':prop.value})

#exchange_options = list(np.unique(code_df['exchange']))
#exchange = Select(title='ExchangeCorrelation', value=exchange_options[0], options=exchange_options)
#exchange.on_change('value', update)
#exchange_df = crossfilter_by_tag(elem_df, {'property':prop.value})

#prop = Select(title='Property', value='B', options=properties)
#prop.on_change('value', update)ppl


apply_crossfilter = Button(label='CrossFilter and Plot')
apply_crossfilter.on_click(CF.update_crossfilter)

clean_crossfilter = Button(label='Clear')
clean_crossfilter.on_click(CF.clear_crossfilter)
# The plotter widgets

x = Select(title='X-Axis', value='k-point', options=plottables)
x.on_change('value', lambda attr, old, new: CF.update_x())

y = Select(title='Y-Axis', value='value', options=plottables)
y.on_change('value', lambda attr, old, new: CF.update_y())

analyse_crossfilt = Button(label='PadeAnalysis')
analyse_crossfilt.on_click(CF.analysis_callback)

elem_df = CF.crossfilter_by_tag(df_obs, {'element':element.value})
code_df = CF.crossfilter_by_tag(elem_df, {'code':code.value})
exchange_df = CF.crossfilter_by_tag(code_df, {'exchange':exchange.value})
struct_df = CF.crossfilter_by_tag(exchange_df, {'structure':struct.value})
prop_df = CF.crossfilter_by_tag(struct_df, {'property':prop.value})

CF_init = CrossFiltDFs(struct_df,elem_df,prop_df)

print ('executed till here')

#z = Select(title='Z-Axis', value='None', options=plottables)
#z.on_change('value', update)



#size = Select(title='Size', value='None', options=['None'] )
#size.on_change('value', update)

#color = Select(title='Color', value='None', options=['None'] )
#color.on_change('value', update)

controls = widgetbox([element, code, exchange, struct, prop, x, y, apply_crossfilter, analyse_crossfilt, clean_crossfilter], width=200)
print ('Initial init figure data', type(CF_init.prop_df))
layout = column(description, ptable, controls, CF_init.create_figure(CF_init.prop_df))

curdoc().add_root(layout)
curdoc().title = "DFT Benchmark"

update()
