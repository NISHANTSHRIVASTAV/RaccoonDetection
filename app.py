import os
from flask import Flask, redirect, url_for, request, make_response, send_file, jsonify
from RaccoonDetection import RaccoonDetector

APP_HOST = "0.0.0.0"
APP_PORT = '5000'
RACCOON_DETECTION_IMAGES = 'results/'

app = Flask(__name__)

@app.route('/ping')
def ping():
    return 'pong'

@app.route('/detect',methods = ['POST'])
def detect():
    
    try:
        if 'nmsThreshold' in request.headers:
            nmsThreshold = request.headers.get('nmsThreshold')
        else:
            nmsThreshold = 0.5
            
        if 'netType' in request.headers:
            netType = request.headers.get('netType')
        else:
            netType = 'mb1-ssd'

        if 'deviceType' in request.headers:
            deviceType = request.headers.get('deviceType')
        else:
            deviceType = 'cpu'

        detector.image = request.files['file'].read()
        detector.nms_threshold = float(nmsThreshold)
        detector.net_type = str(netType)
        detector.device = str(deviceType)
        response = detector.detect_objects()
        return jsonify(response)
    except Exception as e:
        print("Caught an exception in /detect endpoint", e)
    
@app.route('/detect_batch',methods = ['POST'])
def detect_batch():

    try:
        if 'nmsThreshold' in request.headers:
            nmsThreshold = request.headers.get('nmsThreshold')
        else:
            nmsThreshold = 0.5
        
        if 'netType' in request.headers:
            netType = request.headers.get('netType')
        else:
            netType = 'mb1-ssd'

        if 'deviceType' in request.headers:
            deviceType = request.headers.get('deviceType')
        else:
            deviceType = 'cpu'

        files = request.files.getlist("files")
        batch_result = []
        for file in files:
            detector.image = file.read()
            detector.nms_threshold = float(nmsThreshold)
            detector.net_type = str(netType)
            detector.device = str(deviceType)
            response = detector.detect_objects()
            response['input_image'] = str(file.filename)
            batch_result.append(response)
        
        return jsonify({'status':'ok', 'predictions':batch_result, 'error':False, 'batch_size': len(batch_result)})
    except Exception as e:
        print("Caught an exception in /detect_batch endpoint", e)

@app.errorhandler(404)
def page_not_found(e):
    return "Page not found"

@app.route('/get/<filename>', methods=["GET"])
def getfile(filename):

    try:
        file_path = RACCOON_DETECTION_IMAGES + filename

        if os.path.exists(file_path):
            return make_response(send_file(file_path, attachment_filename = filename, add_etags = False, cache_timeout = 0))
        else:
            return "404"
    except Exception as e:
        print("Caught an exception in /get endpoint", e)

if __name__ == '__main__':

    try:   
        print("Initializing Objects..")
        detector = RaccoonDetector()
        print("Starting App..")
        app.run(host = APP_HOST, port = APP_PORT, debug = True, use_reloader = True)
    except Exception as e:
        print("Caught an Exception in main function: ", e)