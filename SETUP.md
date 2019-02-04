# Setting up dft-crossfilter

To setup this platform we recommend installing an environment management
system. For this purpose we installed [anaconda](https://www.continuum.io/downloads). Then create and load an
environment where we will install all the paltform dependencies.

    $ conda create -n dft-crossfilter python=2.7 anaconda

We recommand that you have three terminal windows to run these following
sections. Each will producing a trace in the terminal that you might want
to look into. All three of them are servers and will not return the
terminal input unless you detach them with '&'.  Every section commands
should be run after activating the environment we just created.

## The database setup

You will have to build the database models and run a mongodb instance.
Note that if you have an instance running already you just need the
database models. In case you don't have one you need to get the absolute
path to the data folder in benchmark-db. This is required for storing
the database file and mongod only takes absolutepaths.

    $ source activate dft-crossfilter
    $ cd dft-crossfilter/benchmark-db
    $ pip install -r requirements.txt
    $ python setup.py install

Go to [Mongodb](https://docs.mongodb.com/manual/installation/) and install
it on your system. Then start it. In ubuntu you can do so by typing:

    $ sudo service mongodb start

At this point you should have a mongodb instance running.

## The api setup

After starting the database, we now need to install the api dependencies.
Then run it.

    $ source activate dft-crossfilter
    $ cd dtf-crossfilter/benchmark-api
    $ pip install -r requirements.txt
    $ python run.py --host 0.0.0.0 --port 7000

In a browser go to [API data entry](http://0.0.0.0:7000/bench/push/csv).
This is the api frontend for uploading the dft data. Click on 'Choose File'
and navigate to: dft-crossfilter/benchmark/data/francesca_data_full.csv.
This will push this dft data set into the mongodb database 'benchmark-production'.
To test it out go to [Data description dictionnary](http://0.0.0.0:7000/bench/desc/all).

## bokeh setup

(under testing but this should work, it worked on anaconda on windows)

    $ conda install bokeh
    $ cd benchmark-view
    $ conda install -c r r-essentials
    $ R
    $ > install.packages('minpack.lm')
    close R
    $ bokeh serve crossfilter_app --show

This should automatically start a browser window with the app rendered on it, this can be
host - port tuned later to serve on nginx



(old setup which may not be necessary now)
At this point you are set for the data access part. For the visualizatio part
you will have to build this modified bokeh snapshot. You will need to have gulp
installed. When asked, select the full install with the option:
1) build and install fresh BokehJS

    $ source activate dft-crossfilter
    $ cd dtf-crossfilter/bokeh/bokehjs/
    $ conda install nodejs==4.4.1
    $ sudo apt-get install npm node
    $ sudo apt-get install nodejs-legacy
    $ sudo npm uninstall -g gulp
    $ sudo npm install -g gulp@3.9.1
    $ sudo npm install
    $ sudo gulp build
    $ sudo npm install
    $ cd ..
    $ sudo python setup.py install --build_js
    $ cd ..

If an error occurs we recommand you removing bokehjs/nodes_modules and bokehjs/build.
Now that we have bokeh built with the crossfilter module we can now run our
bokeh visualization server:

    $ python bokeh/bokeh-server --script server.py

Finally, in a browser go to: [Dft-Crossfilter Frontend](http://127.0.0.1:5006/bokeh/benchmark/).

Refer to [Usage](APP_USAGE.md) for a walkthrough of the usage workflow of this tool.
