from flask import Flask, render_template, Response, jsonify
from camera import VideoCamera

app = Flask(__name__)

# Singleton camera instance to share state between video feed and API
camera = VideoCamera()

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def get_status():
    return jsonify(camera.current_status)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
