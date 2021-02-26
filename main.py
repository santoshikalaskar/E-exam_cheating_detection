from flask import Flask, render_template, Response
from Face_movement_detection import Face_Movement_Detection

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.Start_Web_Cam()
        if frame != None:
            yield (b'--frame\r\n' + b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        if frame == None:
            return 0

@app.route('/video_feed')
def video_feed():
    # Face_Movement_Detection().Get_Training_Data()
    return Response(gen(Face_Movement_Detection()),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0',port='5000', debug=True)
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
