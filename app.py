from app import create_app
import os

app = create_app()

from ml.ml_api import ML_Voice_Generate
from flask import send_file
@app.route('/ml/voice', methods=['POST'])
def ml_voice():
    ML_Voice_Generate().post()
    
    print("done")
    return send_file("/home/ubuntu/Real-Time-Voice-Cloning/test_file.wav"), 200
    

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', threaded=True)
