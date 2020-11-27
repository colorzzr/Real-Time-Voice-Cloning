from flask_restx import Resource
from flask import request
from bson.objectid import ObjectId
import inflect
import math
import translators as ts

# from app import db_connection


# this api instance is make random number of recipe for front page
class Translator(Resource):

    def post(self):
        post_data = request.get_json()
        input_text = post_data.get('text', None)

        text = ts.google(input_text)

        return {'result':text}, 200
