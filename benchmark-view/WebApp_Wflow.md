Clean up of dft benchmark data and integration for database deployment and readiness 

* MainCollectionFiles:
  contains database csv that is easily transferrable to MongoDB
  - 2 collections: Main Calculated data, Ref Data 
  - needs to be updatable with a plug and play script 

* AutoScripts:
  Workshop for automatic plotting scripts, analysis scripts that takes 
  a crossfiltered selection as input. Bring down to 2 scripts or single Class
  - database ccrossfilter script based on UI widget inputs 
  - script that commands the analysis of the crossfiltered data
  - goto scripts for commanding a high throughput work through of new data that 
    can be updated into an input collection.  

* Static_Image_examples:
  Example static files of plots that can be displayed on the front page 
  as examples, statistics. 

* RefStdFiles:
  Files that hold experimental references for calculation of accuracies. Second

