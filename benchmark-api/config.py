import os
basedir = os.path.abspath(os.path.dirname(__file__))

WTF_CSRF_ENABLED = True
SECRET_KEY = '33stanlake#'

DEBUG = True

APP_TITLE = 'Benchmark API'

VERSION = '0.1-dev'

MONGODB_SETTINGS = {
    'db': 'benchmark-production',
    'host': '0.0.0.0',
    'port': 27017
}