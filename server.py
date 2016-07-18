from bokeh.plot_object import PlotObject
from bokeh.server.utils.plugins import object_page
from bokeh.server.app import bokeh_app
from bokeh.plotting import curdoc, cursession
from bokeh.crossfilter.models import CrossFilter

from benchmark.loader import load

@bokeh_app.route("/bokeh/benchmark/")
@object_page("benchmark")
def make_crossfilter():
	"""The root crossfilter controller"""
	# Loading the dft data head as a 
	# pandas dataframe
    autompg = load("francesca_data_head")
    app = CrossFilter.create(df=autompg)
    return app
