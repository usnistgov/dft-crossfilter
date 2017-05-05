import datetime
from ..core import db
import json
from bson import ObjectId
          
class Attribute(db.Document):
    created_at = db.DateTimeField(default=datetime.datetime.utcnow())
    name = db.StringField()
    values = db.ListField()

    def clone(self):
        del self.__dict__['_id']
        del self.__dict__['_created']
        del self.__dict__['_changed_fields']
        self.id = ObjectId()

    def info(self):
        data = {'name':self.name, 'values':self.values}
        return data

    def to_json(self):
        data = self.info()
        return json.dumps(data, sort_keys=True, indent=4, separators=(',', ': '))