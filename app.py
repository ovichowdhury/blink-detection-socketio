from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit, send
from flask_cors import CORS
from face_lib import is_eye_open
from expression_lib import detectExpression

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")


@socketio.on('connect')
def on_connect():
    print("User connected: ", request.sid)
    

@socketio.on('disconnect')
def on_disconnect():
    print('Client disconnected')


@socketio.on('is_eye_open')
def on_eye_open(json):
    # print(json)
    try:
        image_base64 = json['image']
        eye_open_status = is_eye_open(image_base64)
        print(eye_open_status)
        emit('is_eye_open_res', eye_open_status)
    except Exception as ex:
        print(ex)


@socketio.on('detect_expr')
def on_detect_expr(json):
    try:
        image_base64 = json['image']
        expr = detectExpression(image_base64)
        print(expr)
        emit('detect_expr_res', expr)
    except Exception as ex:
        print(ex)

if __name__ == '__main__':
    print("Web Socket Service Running...")
    socketio.run(app, host='0.0.0.0', port=5001)