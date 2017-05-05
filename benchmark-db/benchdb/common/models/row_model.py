import datetime
from ..core import db
import json
from bson import ObjectId
          
class Row(db.Document):
    created_at = db.DateTimeField(default=datetime.datetime.utcnow())
    bz_integration = db.StringField()
    calculations_type = db.StringField()
    code = db.StringField()
    element = db.StringField()
    exchange = db.StringField()
    extrapolate = db.FloatField()
    extrapolate_err = db.FloatField()
    k_point = db.FloatField()
    pade_order = db.FloatField()
    perc_precisions = db.FloatField()
    precision = db.FloatField()
    property = db.FloatField()
    structure = db.StringField()
    value = db.FloatField()

    def clone(self):
        del self.__dict__['_id']
        del self.__dict__['_created']
        del self.__dict__['_changed_fields']
        self.id = ObjectId()

    def info(self):
        data = {'bz_integration':self.bz_integration, 'calculations_type':self.calculations_type}
        data['code'] = self.code
        data['element'] = self.element
        data['exchange'] = self.exchange
        data['extrapolate'] = self.extrapolate
        data['extrapolate_err'] = self.extrapolate_err
        data['k-point'] = self.k_point
        data['pade_order'] = self.pade_order
        data['perc_precisions'] = self.perc_precisions
        data['precision'] = self.precision
        data['property'] = self.property
        data['structure'] = self.structure
        data['value'] = self.value
        return data

    def to_json(self):
        data = self.info()
        return json.dumps(data, sort_keys=True, indent=4, separators=(',', ': '))