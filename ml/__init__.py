from app import module_api
from .ml_api import ML_Fine_Tune, ML_Voice_Generate, Translator_Api, Get_file, save_file
from .user_api import check_user_repo

# create the recipe namespace
ml_ns = module_api.namespace(name='translator_api', path='/v1/ml')

# add recipe related api
ml_ns.add_resource(save_file, '/file/save')
ml_ns.add_resource(Get_file, '/users/<username>/file/<filename>')
ml_ns.add_resource(Translator_Api, '/translator')
ml_ns.add_resource(ML_Fine_Tune, '/users/<username>/fine-tuning')


ml_ns.add_resource(check_user_repo, '/users/<username>')
