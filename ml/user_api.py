from flask_restx import Resource
from flask import request, send_from_directory
import os
from os.path import isfile, join
from shutil import copyfile

# the function will check if user exist under
# if user is first login then create the folder
class check_user_repo(Resource):
    def put(self, username):
        username = username.replace(' ', '')

        if not os.path.exists('user_data/generated_voice/'+username):
            # make user dir
            os.mkdir('user_data/generated_voice/'+username)
            os.mkdir('user_data/recordings/'+username)
            os.mkdir('user_data/favorite/'+username)

        return {'result':'success'}, 200


# get api will get all save file under the users' favori
# the post will allow user to add generate voice for later usage
# the delete action will remove the recording sentense
class favorite_recording(Resource):
    def get(self, username):
        # post_data = request.get_json()

        # get saved file under it
        try:
            favorite_path = 'user_data/favorite/'+username
            saved_audios = []
            # print(favorite_path)

            for (dirpath, dirnames, filenames) in os.walk(favorite_path):
                saved_audios += filenames
                # print(filenames)

            ret_file = []
            for f in saved_audios:
                ret_file.append(f[:-4].replace("_", " "))

        except Exception as e:
            return {'result': e}, 400

        return {'result':ret_file}, 200

    def post(self, username):
        post_data = request.get_json()

        # get the name of generated audio
        # assume it is under the user_data/generated_voice/<username>
        target_audio = post_data.get("filename")
        sentense = post_data.get("text").replace(" ", "_")

        #copy the file from generated_voice into new favorite folder
        try:
            src_folder = 'user_data/generated_voice/'+username + '/' + target_audio
            dst_folder = 'user_data/favorite/'+username + '/' + sentense + '.wav'
            copyfile(src_folder, dst_folder)
        except Exception as e:
            return {'result': e}, 400

        return {'result':'success'}, 200


    def delete(self, username):
        post_data = request.get_json()

        # get the name of generated audio
        # assume it is under the user_data/generated_voice/<username>
        removed_audio = post_data.get("delete_text").replace(" ", "_") + '.wav'
        

        #copy the file from generated_voice into new favorite folder
        try:
            os.remove('user_data/favorite/'+username + '/' + removed_audio)
        except Exception as e:
            return {'result': e}, 400

        return {'result':'success'}, 200