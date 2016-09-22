from wtforms import Form, BooleanField, StringField, validators, \
ListWidget
import requests
import os

from pymongo import MongoClient

### check necessity of this 
if DEBUG:
    host = '10.5.46.101'
else:
    host = 'localhost'


class SelectNISTdata(Form):
    """
    form page python object
    that creates the NIST analysis
    home page data 
    ** features required for sure:
     - drop down list selection of data attributes for
       ## implement as String field with validator till we figure out
          correct input dtype (Id error, is it not a list? what other iterable?) 
          for the ListWidget
        - code 
        - element(s)
        - property ..
        - functional 
        - integration method 
        ** objective: create a filtered data query for a list of 
                      elements with the same attributes 
     - display the slected data in some pleasing tabular form 
       basically print the filtered out dataframe with the relevant 
       features. 
    * extension methods:
        - multiple selection into the filtered data set 
 
    """
    #### Simple boxes
    ## code seletion field defined as an empty field
    # validators can be direct text matching also
    code_b = StringField('', validators.Length( min=4, max=5, message=u'Check input') ) 
    element_b = StringField('', validators.Length( min=1, max=2, message=u'Enter Al, W, Ni, Fe or Cu') )
    ####

    #### test .. the field input to ListWidget must be a kwargs of fields like above 
    ## LIST box 1 : code selection widget
    vasp = StringField('VASP')
    Dmol3 = StringField('DMol3')
    PWscf = StringField('PWscf')
    code = ListWidget()
    code(vasp,Dmol3,PWscf)
    ##
    ## box 2: element selection widget  (means this code needs to be updated when more elements calculated
    Al = StringField('Al')
    W = StringField('W')
    Ni = StringField('Ni')
    Fe = StringField('Fe')
    Cu = StringField('Cu')
    elements = ListWidget()
    elements(Al, W, Ni, Fe, Cu)
    ##
    ####
    B = StringField('Bulk Modulus')
    dB = StringField('Bulk Modulus derivative')
    E0 = StringField('DFT Ground state Energy Minimum')
    a0 = StringField('Lattice constant')
    properties = ListWidget()
    properties(B, dB, E0, a0)

    ### test for table result .. communuication with the API should have been made 

def Render_select_form_query(request):
    """
    html renderer based on user's selections of defined 
    attributes: code, property, elements
    for user to select code on nist_query_home.html 

    written based on WTforms examples
    """
#    user = request.current_user ##may not be needed according to exmaples
    data_tags = {} 
    form = SelectNISTdata(request.POST) 
    if request.method == 'POST' and form.validate():        
        user = User() ## check if this is class from python requests or it needs to be subclassed
        user.code = form.code.data
        user.elements = form.elements.data
        user.properties = form.properties.data
        user.save()    
        redirect('nist_query_home')
    return render_response('nist_query_home.html')

def Query_REST_API():
    """
    query the REST API for data and return a filtered dataframe to be 
    rendered
    """
    ## to do to replace mimicing what bokeh server does with the API 
    pass


def loadNIST(filename=None, query={}):
    """
    loads all NIST data from the NIST collection on 
    mongoDB 
    Filename provided as temporary testing in place
    of DB 
    converts database/csv data to pandas dataframe
    """
    if filename:
       dat = pd.read_csv('./temp_data'+os.sep+filename)
    elif query:
       dat = pd.from_json(Query_Data(query))   
    return dat


def filterNIST_data(df, tags):
    """
    df: starting dataframe
    tags: dict of identifiers ['column_name':identifying field]
    """
    filtered_df = df
    for c,i in tags.items():
        filtered_df = filtered_df[filtered_df[c]==i]
    return filtered_df


def Query_Data():
    """
    standby to test querying on a csv file or the MongoDB through
    MongoClient
    """
    mongo_engine = MongoClient(host=host, port=27017)
    db = mongo_engine['vasp']
    if db.authenticate(user=USR, password=PWD):
        collection = db.NIST
    return collection.find(query)
    pass

def Render_display_query_result(request):
    """
    render the query result page utilizing a TableWidget? 
    """
    pass
    user = request.current_user
    form = SelectNISTdata(request.POST,user)
    form.populate_obj(user)
    redirect('nist_query_home') 
    return render_response('nist_query_home.html')


def call_Shiny(request):
    ## call shiny app based on user choice, should render and open another 
    ## window that opens up on the browser 
    ## contains choices of Shiny Apps as a RadioButton form widget
    pass

if __name__=='__main__':
    # first render the html home page with the querying widgets
    data_query, _ = Render_select_form_query()
    # page with widgets is rendererd and data query tag is returned
    dat = loadNIST(query=data_query)  ## stage 1 query done on database/API/csv 
                                      # test file
    ## optionally call the filteredNIST here ? 
    # operate on the queried data (dat) and call Shiny on it based on the user's choice by an action click
    call_Shiny()





