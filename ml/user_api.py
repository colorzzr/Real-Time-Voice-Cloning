from flask_restx import Resource
from flask import request, send_from_directory
import os

# the function will check if user exist under
# if user is first login then create the folder
class check_user_repo(Resource):
    def put(self, username):
        username = username.replace(' ', '')

        if not os.path.exists('user_data/generated_voice/'+username):
            # make user dir
            os.mkdir('user_data/generated_voice/'+username)
            os.mkdir('user_data/recordings/'+username)

        return {'result':'success'}, 200

