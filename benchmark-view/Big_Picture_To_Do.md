What we want to do in this branch:

  "create a flask app using the new bokeh server that runs on Apache and can render an Iframe from Shiny R"

    - which uses bokeh's crossfilter model classes (that makes use of pandas dataframe tools)
      to crossfilter data
    - use the new bokeh server to interact directly with the REST api of benchmark-db
    - use Shiny R to create html that can be Iframed into python served bokeh app for the data analysis.
    - this bokeh app can be wrapped around with an Apache or Nginx server.


  ** User interface goals
    - User sees at /home.html a website that looks like https://materialsweb.org/nist_page
      - the webpage itself can have a periodic table type interactive interface or a library website like
        form with drop down lists or text entry elements
      - for now the UI can directly be the crossfilter UI of click and drag
      - data crossfiltered into the UI, if it can be statistically analyzed can be launched separately
        - by if statistically analyzed means if the data is within dimensions accessible by the statistical tools
          For example: 2D data can be fit using regression tools
          ** provide a link to examples of data crossfiltered correctly so that it can be analyzed with the
             statistics.
      - an about page linked that summarizes the project
      - a contact page linked that summarizes whom to contact

  ** widgets
     - plotting checkboxes (log ? )
     - zoom/pan/download
     - text entry of queries
