from app import module_api
from .ml_api import ML_Fine_Tune, ML_Voice_Generate

# create the recipe namespace
ml_ns = module_api.namespace(name='translator_api', path='/v1/ml')

# add recipe related api
ml_ns.add_resource(ML_Fine_Tune, '/fine-tuning')
ml_ns.add_resource(ML_Voice_Generate, '/voice')
