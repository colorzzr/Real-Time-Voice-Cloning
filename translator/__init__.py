from app import module_api
from .translator_api import Translator

# create the recipe namespace
ts_ns = module_api.namespace(name='translator_api', path='/v1/translator')

# add recipe related api
ts_ns.add_resource(Translator, '/')
