import os
from os.path import dirname, join

from collections import OrderedDict
import pandas as pd
import numpy as np

from bokeh.io import curdoc
from bokeh.layouts import row, widgetbox, column
from bokeh.models import Select, Div, Column, HoverTool, ColumnDataSource, Button, CheckboxButtonGroup
from bokeh.plotting import figure
from bokeh.sampledata.periodic_table import elements


df_obs = pd.read_csv('./crossfilter_app/Data/DataC.csv')
print ('read df_obs')

# single reference standard this can be an on request
# basis input as well
#df_ref = pd.read_json('./Data/Ref.json')

# dividing data into gneeral discretes and continuous

columns = sorted(df_obs.columns) #+ sorted(df.columns)
discrete = [x for x in columns if df_obs[x].dtype == object] 

continuous = [x for x in columns if x not in discrete]

####################################################################

##divide data into plottables and non plottables (aggregates or 3D plottables) ,
#keep it to 2D plottables for now, this is known from the column names themselves

plottables =  ['k-point', 'value', 'perc_precisions']

x_select = Select(title='X-Axis', value='k-point', options=plottables)

y_select = Select(title='Y-Axis', value='value', options=plottables)

non_plottables = [ x for x in columns if x not in plottables ] # for aggregates

structures = list(np.unique(df_obs['structure']))
_elements = list(np.unique(df_obs['element']))
#print (_elements)
exchanges = list(np.unique(df_obs['exchange']))
properties = list(np.unique(df_obs['property']))
codes = list(np.unique(df_obs['code']))

# which sets of k-point and value to string together ? any unit transformations on the dataset values or k-point

## have another dataframe (mongo collection) for the reference standards to compute the accuracy (uniquely identified by the element SAME standard should apply to all codes/exchanges/elements.

############## Header Content from description.html  #################

content_filename = join(dirname(__file__), "test_desc.html")

description = Div(text=open(content_filename).read(),
                  render_as_text=False, width=600)


# periodic table widget
romans = ["I", "II", "III", "IV", "V", "VI", "VII"]

elements["atomic mass"] = elements["atomic mass"].astype(str)

print("Table---")
#print(elements.period)
print("---Table")
try:
  elements["period"] = [romans[x-1] for x in elements.period]
except:
  pass
elements = elements[elements.group != "-"]

group_range = [str(x) for x in range(1, 19)]


colormap = {
    "c"        : "#ffa07a",
    "nc"       : "#A9A9A9"
}

elems_colorpair = {'H':'nc','He':'nc',
                   'Li':'nc','Be':'nc','B':'nc','C':'nc', 'N':'nc', 'O':'nc','F':'nc','Ne':'nc',
                   'Na':'nc','Mg':'nc', 'Al':'c','Si':'nc','P':'nc','S':'nc','Cl':'nc','Ar':'nc',
                   'K': 'nc', 'Ca':'nc','Sc':'c', 'Ti':'c' ,'V':'c' , 'Cr':'c', 'Mn':'c', 'Fe':'c', 'Co':'c', 'Ni':'c', 'Cu':'c', 'Zn':'c',
                   'Rb':'nc', 'Sr':'nc','Y':'c', 'Zr':'c', 'Nb':'c', 'Mo':'c', 'Tc':'c', 'Ru':'c', 'Rh':'c', 'Pd':'c', 'Ag':'c','Cd': 'c',
                   'Cs':'nc', 'Ba':'nc', 'Hf':'c', 'Ta':'c', 'W':'c', 'Re':'c', 'Os':'c', 'Ir':'c', 'Pt':'c', 'Au':'c', 'Hg':'c'
                 }
elems_colorpair.update( { key:'nc' for key in list(elements['symbol']) if key not in list(elems_colorpair.keys()) } )


print ([ colormap[elems_colorpair[x]] for x in elements['symbol'] ])

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
#        cpk=elements["CPK"],
        atomic_number=elements["atomic number"],
#        electronic=elements["electronic configuration"],
#        mass=elements["atomic mass"],
        B=['B' for x in elements["atomic mass"]],
        dB=['dB' for x in elements["atomic mass"]],
        V0=['V0' for x in elements["atomic mass"]],
        E0=['E0' for x in elements["atomic mass"]],
#        type=elements["metal"],
        type_color=[ colormap[elems_colorpair[x]] for x in elements['symbol'] ],
    )
)

# plot the periodic layout
name = source.data["name"]
B = source.data["B"]

ptable = figure(title="Periodic Table", tools="hover",
           x_range=group_range, y_range=list(reversed(romans)))

ptable.plot_width = 1500
ptable.toolbar_location = None
ptable.outline_line_color = None

ptable.background_fill_color = 'white'
ptable.rect("group", "period", 0.9, 0.9, source=source,
       fill_alpha=0.3, color='type_color')

text_props = {
    "source": source,
    "angle": 0,
    "color": "black",
    "text_align": "left",
    "text_baseline": "middle"
}

ptable.text(x="symx", y="period", text="sym",
       text_font_style="bold", text_font_size="22pt", **text_props)

ptable.text(x="symx", y="numbery", text="atomic_number",
       text_font_size="9pt", **text_props)

ptable.grid.grid_line_color = None


ptable.select_one(HoverTool).tooltips = [
    ("name", "@name"),
    ("V0 (A^3 per atom)", "@V0"),
    ("B (GPa)", "@B"),
    ("dB/dP", "@dB")
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


    def update_ptable(self):
        """
        update the periodic table highlighted elements
        """
        from bokeh.sampledata.periodic_table import elements
        romans = ["I", "II", "III", "IV", "V", "VI", "VII"]

        elements["atomic mass"] = elements["atomic mass"].astype(str)

        elements["period"] = [x for x in elements.period]
        elements = elements[elements.group != "-"]

        group_range = [str(x) for x in range(1, 19)]
        print ('reaches colormap def')
        colormap = {
                     "c"        : "#ffa07a",
                     "nc"       : "#A9A9A9"
                   }
        elems_colorpair = {}

        B_extrapol_props = {}
        dB_extrapol_props = {}
        V0_extrapol_props = {}
        E0_extrapol_props = {}

        for e in elements["symbol"]:
            for p in np.unique(list(self.struct_df['property'])):
               if e in np.unique(list(self.struct_df['element'])):
                 #print (p,e,'avail')
                 e_struct = self.struct_df[self.struct_df['element']==e]
                 p_e_struct = e_struct[e_struct['property']==p]
                 elem_prop = {e: np.unique(list(p_e_struct['extrapolate']))[0]}
               else:
                 elem_prop = {e:'xxx'}

               if p=='B':
                 B_extrapol_props.update(elem_prop)
               elif p=='dB':
                 dB_extrapol_props.update(elem_prop)
               elif p=='v0':
                 print ('V0', elem_prop)
                 V0_extrapol_props.update(elem_prop)
               elif p =='E0':
                 E0_extrapol_props.update(elem_prop)
                 
        elems_colorpair.update( { key:'c' for key in np.unique(list(self.struct_df['element'])) } )
        elems_colorpair.update( { key:'nc' for key in list(elements['symbol']) if key not in list(elems_colorpair.keys()) } )


        print ([ colormap[elems_colorpair[x]] for x in elements['symbol'] ])

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
#                     cpk=elements["CPK"],
                     atomic_number=elements["atomic number"],
#                     electronic=elements["electronic configuration"],
                     B=[B_extrapol_props[x] for x in elements["symbol"]],
                     dB=[dB_extrapol_props[x] for x in elements["symbol"]],
                     V0=[V0_extrapol_props[x] for x in elements["symbol"]],
                     E0=[E0_extrapol_props[x] for x in elements["symbol"]],
                     type=elements["metal"],
                     type_color=[ colormap[elems_colorpair[x]] for x in elements['symbol'] ],
                      )
                                 )

        # plot the periodic layout
        #name = source.data["name"]
        #B = source.data["B"]

        ptable = figure(title="Periodic Table", tools="hover",
           x_range=group_range, y_range=list(reversed(romans)))
        ptable.background_fill_color='white'
        ptable.plot_width = 1500
        ptable.toolbar_location = None
        ptable.outline_line_color = None

        ptable.rect("group", "period", 0.9, 0.9, source=source,
                    fill_alpha=0.3, color='type_color')

        text_props = {
           "source": source,
           "angle": 0,
           "color": "black",
           "text_align": "left",
           "text_baseline": "middle"
                     }
 
        ptable.text(x="symx", y="period", text="sym",
        text_font_style="bold", text_font_size="22pt", **text_props)

        ptable.text(x="symx", y="numbery", text="atomic_number",
        text_font_size="9pt", **text_props)

#        ptable.text(x="symx", y="namey", text="name",
#        text_font_size="6pt", **text_props)

#        ptable.text(x="symx", y="massy", text="mass",
#        text_font_size="5pt", **text_props)

        ptable.grid.grid_line_color = None


        ptable.select_one(HoverTool).tooltips = [
        ("name", "@name"),
        ("V0 (A^3 per atom)", "@V0"),
        ("B (GPa)", "@B"),
        ("dB/dP", "@dB")]
        return ptable
    
    def create_figure(self,dataset,datplot='Init',plot_type=None):
        """
        figure and plot creation for a given dataset
        TODO: enable support for multiple selection
        refactor to a simple figure creator and
        add helper functions for the plots
        """
        kw = dict()

        x_title = x_select.value.title() + ' Density per atom'
        
        # hack for labels now 

        if isinstance(dataset,pd.DataFrame):
          if np.unique(list(dataset['property']))[0]=='B':
             y_title = 'Bulk Modulus (GPa) '+y_select.value.title()
          elif np.unique(list(dataset['property']))[0]=='dB':
             y_title = 'dB/dP '+y_select.value.title()
          elif np.unique(list(dataset['property']))[0]=='v0':
             y_title = 'Volume per atom (A^3) '+y_select.value.title()
          elif np.unique(list(dataset['property']))[0]=='E0':
             y_title = 'DFT Energy per atom (eV/atom) '+y_select.value.title()
        else:
             y_title = 'Pade Prediction'

        kw['title'] = "%s vs %s" % (y_title, x_title)

        #if x_select.value=='k-point':
        kw['x_axis_type'] = 'log'

        if x_select.value == 'perc_precisions' and y_select.value == 'perc_precisions':
          kw['y_axis_type'] = 'log'

        self.p = figure(plot_height=600, plot_width=800, tools='pan,wheel_zoom,box_zoom,reset,hover', **kw)

        # sets the axes
        self.p.xaxis.axis_label = x_title
        self.p.yaxis.axis_label = y_title


        if x_select.value in continuous:
          self.p.xaxis.major_label_orientation = pd.np.pi / 4


        #print (dataset)
        if datplot=='Init':
           # if data is to be plotted
           xs =dataset[x_select.value].values
           ys = dataset[y_select.value].values
           self.xs_init = xs
           self.ys_init = ys

           self.p.scatter(x=xs, y=ys)#, alpha=1.0, hover_color='blue', hover_alpha=1.0)
           return self.p

        elif datplot == 'Add':
           # add a plot to figure, from statistical analysis
           if plot_type == 'plot_pade':

               #pade_order = self.analysis_results['Order']
               #pade_extrapolate = self.analysis_results['Extrapolate']
               #print (pade_extrapolate, float(pade_extrapolate))

               # create precisions based on the extrapolate
               #print (self.add_data)
               xs = self.add_data[0]
               ys = self.add_data[1]#[abs(y-pade_extrapolate) for y in self.ys_init]
               #print (ys)
               # print (xs,ys,len(xs),len(ys))
               print ("Plots a line supposedly")
               #print (len(self.ys_init), len(ys))
               #l = min([len(self.ys_init), len(ys), len(self.xs_init),len(xs)])
               #self.plot_layout.scatter(x=self.xs_init[0:l], y=self.ys_init[0:l])#, alpha=1.0, hover_color='blue', hover_alpha=1.0)
               #print (type(self.plot_layout))
               #self.p.self.plot
               self.p = figure(plot_height=600, plot_width=800, tools='pan,wheel_zoom,box_zoom,reset,box_zoom, hover', **kw)
               print('executes till re-figure')
               self.p.circle(x=self.xs_init,y=self.ys_init)
               print('executes till circle')
               self.p.line(x=xs, y=ys, line_color='red')
               #self.p.line_color='red'
               print('executes till line')
               return self.p

        

        else:
          # clear the figure by plotting an empty figure
          xs = []
          ys = []
          self.p = figure(plot_height=600, plot_width=800, tools='pan,wheel_zoom,box_zoom,reset,hover', **kw)
          self.p.scatter(x=xs, y=ys)#, alpha=1.0, hover_color='blue', hover_alpha=1.0)
          return self.p

    # The crossfilter widgets
    def update(self, attr, old, new):
       print ('Attribute', attr, 'OLD', old, 'NEW', new)
       #print (len(layout.children))
       print ('executes here on update')#, exchange_df)

    def update_code(self):
        """
        update for the code selection
        """
        print ('update code')
        self.code_df = df_obs[df_obs['code'] == code.value].dropna()

    def update_exchange(self):
        """
        update the exchange
        """
        print ('update exchange')
        self.exchange_df = self.code_df[self.code_df['exchange']== exchange.value].dropna()

    def update_element(self,new):
        print ('Updating element down selection for property',element.active[0])
        self.elem_df = self.struct_df[self.struct_df['element'] == _elements[element.active[0]] ].dropna()
        self.plot_data = self.elem_df

    def update_struct(self):
       #print ('Updating struct down selection for element')
       #print ("struct.value",struct.value)
       self.struct_df = self.exchange_df[self.exchange_df['structure'] == struct.value].dropna()
       print ('Updating ptable with structure selection')
       layout.children[2] =  self.update_ptable()
       elem_checkbox= CheckboxButtonGroup(labels=np.unique(list(self.struct_df['element'])), active=[1])
       controls2.children[2] = elem_checkbox
       
       self.plot_data = self.struct_df
       print ('finished callback to update layout')


    def update_prop(self):
       #print ('Updating struct down selection for element')
       #print (prop.value)
       self.prop_df = self.elem_df[self.elem_df['property'] == prop.value].dropna()
       #print ('The final dict', self.prop_df.to_dict(orient='list'))
       self.plot_data = self.prop_df

    def update_x(self):
        pass

    def update_y(self):
        pass

    def update_crossfilter(self):
       print ('Triggering crossfilter')
       #print (type(self.plot_data))
       #print (np.unique(self.plot_data['property']))
       layout.children[4] = self.create_figure(self.plot_data)

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
        layout.children[4] = self.create_figure(self.plot_data)

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
        print ('wrote out data file')
        os.system('Rscript crossfilter_app/non_err_weighted_nls.R')
        self.analysis_results = pd.read_csv('crossfilter_app/Result.csv')
        #self.add_data = [ list(self.xs_init), list(self.predict_results['Preds....c.predict.m2..']) ]
        ext_values = list(self.analysis_results['Extrapolate....extrapolates'])
        error_values = list(self.analysis_results['Error....errors'])
        self.ext_min_error = ext_values[error_values.index(min(error_values))]
        print ('executed R script on crossfiltered data')
        if error_values.index(min(error_values))==0:
            self.predict_results = pd.read_csv('crossfilter_app/Pade1.csv')
            self.add_data = [list(self.predict_results['Px....x_plot']), list(self.predict_results['Py....pade1.x_plot.'])]
        elif error_values.index(min(error_values))==1:
            self.predict_results = pd.read_csv('crossfilter_app/Pade2.csv')
            self.add_data = [list(self.predict_results['Px....x_plot']), list(self.predict_results['Py....pade2.x_plot.'])]

        print ('ADD DATA', self.add_data)
        layout.children[4] = self.create_figure(self.add_data, datplot='Add', plot_type='plot_pade')

def update():
    pass
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

element = CheckboxButtonGroup(labels=_elements, active=[1])
element.on_click(CF.update_element)

prop = Select(title='Property', value=properties[0], options=properties)
prop.on_change('value', lambda attr, old, new: CF.update_prop())

apply_crossfilter = Button(label='CrossFilter and Plot')
apply_crossfilter.on_click(CF.update_crossfilter)

clean_crossfilter = Button(label='Clear')
clean_crossfilter.on_click(CF.clear_crossfilter)

x_select.on_change('value', lambda attr, old, new: CF.update_x())

y_select.on_change('value', lambda attr, old, new: CF.update_y())

analyse_crossfilt = Button(label='PadeAnalysis')
analyse_crossfilt.on_click(CF.analysis_callback)

code_df = CF.crossfilter_by_tag(df_obs, {'code':code.value})
exchange_df = CF.crossfilter_by_tag(code_df, {'exchange':exchange.value})
struct_df = CF.crossfilter_by_tag(exchange_df, {'structure':struct.value})
elem_df = CF.crossfilter_by_tag(struct_df, {'element':_elements[0]})
prop_df = CF.crossfilter_by_tag(elem_df, {'property':prop.value})

CF_init = CrossFiltDFs(code_df,exchange_df,struct_df,elem_df,prop_df)

print ('executed till here')

#z = Select(title='Z-Axis', value='None', options=plottables)
#z.on_change('value', update)


controls1 = widgetbox([code, exchange, struct], width=400)
controls2 = widgetbox([element, prop, x_select, y_select, apply_crossfilter, analyse_crossfilt, clean_crossfilter], width=400)
#print ('Initial init figure data', type(CF_init.prop_df))
layout = column(description, controls1, ptable, controls2, CF_init.create_figure(CF_init.prop_df))

curdoc().add_root(layout)
curdoc().title = "DFT Benchmark"

update()
