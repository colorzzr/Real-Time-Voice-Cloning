from flask_restx import Resource
from flask import request
from bson.objectid import ObjectId
import inflect
import math
import translators as ts

# from app import db_connection


# this api instance is make random number of recipe for front page
class ML_Voice_Generate(Resource):

    def post(self):
        post_data = request.get_json()
        input_text = post_data.get('text', 'None')

        text = ts.google(input_text)


        # use the text to and model to get the audio
        # TODO

        return {'result':'ML_Voice_Generate'}, 200



# this api instance is make random number of recipe for front page
class ML_Fine_Tune(Resource):

    def post(self):
        post_data = request.get_json()
        # todo here it should be a file
        input_text = post_data.get('voice', None)

        # use the audio to do the fine tuning
        # TODO

        return {'result':'ML_Fine_Tune'}, 200