import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from flask import Flask, request, jsonify, make_response
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS, cross_origin
import cv2
import os
import math
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
# Allow requests from http://localhost:3000
# CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})
CORS(app)
# CORS(app, resources={r"/api/*": {"origins": "*"}})
# app.config['CORS_HEADERS'] = 'Content-Type'
# cors = CORS(app, resource={
#     r"/*":{
#         "origins":"*"
#     }
# })
# socketio = SocketIO(app)
socketio = SocketIO(app, cors_allowed_origins="*")


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

fielname = "TrainData.csv"
df = pd.read_csv(fielname)
asana_names = df.Asana
filename = 'classifier.sav'
classifier = pickle.load(open(filename, 'rb'))

filename = 'kNN.sav'
kNN = pickle.load(open(filename, 'rb'))


@app.route("/")
# @cross_origin()
def index():
    return jsonify("This is the python backend page")


# @app.route('/msg', methods=['POST', 'OPTIONS'])
# @socketio.on("msg_from_frontend")
# def test(data):
#     if request.method == "OPTIONS":
#         return _build_cors_preflight_response()
#     if request.method == "POST":  # CORS preflight
#         print("data" + data)
#         socketio.emit('msg_from_backend',"Hi frontend")
#     else:
#         raise RuntimeError("I don't know")

# @socketio.on("connection")
# def idk():
#     print("I just wnat to run this to see what is going on")
#     socketio.emit("message", "sup")
#     pass


# @socketio.on("my event")
# def handle_event(string):
#     print("This is my string " + str(string))
#     socketio.emit("message", "Hi frontend")

@socketio.on("connect")
def handle_connect():
    room_id = request.sid  # Use the client's socket ID as the room ID
    print("An user joined in the room id: ", room_id)
    join_room(room_id)

@socketio.on("disconnect")
def handle_disconnect():
    room_id = request.sid
    print("An user left the room id: ", room_id)
    leave_room(room_id)


@socketio.on("videoFrame")
def handle_video_frame(data):
    room_id = request.sid
    try:
        # print("yaha aaya")
        # Access image data from the data
        image_data = data.get("imageData")

        binary_image = base64.b64decode(image_data)

        # Process the image data (you may need additional processing logic here)
        print("Received image data length:", len(binary_image))

        # Convert binary image data to numpy array
        nparr = np.frombuffer(binary_image, np.uint8)

        # print("nparr")
        # print(nparr)

        # Decode the image using OpenCV
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # print("frame")
        # print(frame)

        # Process the image frame
        result = process_single_frame(frame)
        print(result)

        socketio.emit("message", result,room=room_id)
        # socketio.emit("message", "Oye frontend I have received your frames")
        # print("yaha aaya2")

    except Exception as e:
        print("Error handling video frame:", e)


@socketio.on_error_default
def default_error_handler(e):
    print('An error occured', e)


def find_mid(a, b):
    a = np.array(a)  # First
    b = np.array(b)  # Second
    return (a+b)/2


def distance(a, b):
    a = np.array(a)  # First
    b = np.array(b)  # Second
    return (((a[0]-b[0])**2) + ((a[1]-b[1])**2) + ((a[2]-b[2])**2)) ** 0.5


def magnitude(a):
    a = np.array(a)
    return ((a[0]**2)+(a[1]**2)+(a[2]**2))**0.5


def get_line(a, b):
    a = np.array(a)  # First
    b = np.array(b)  # Second
    return a-b


def get_joints(landmarks):
    #     joints = head(mid point of eyes), neck, left shoulder, right shoulder, left elbow, right elbow, left wrist, right wrist, left hip, mid hip, right hip, left knee, right knee, left ankle, right ankle
    joints = []
    # print("landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x")
    # print(landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].x)
    joints.append(find_mid([landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].y, landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].z],
                           [landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].z])
                  )
    joints.append(find_mid([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z],
                           [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z])
                  )
    joints.append(np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]))
    joints.append(np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z]))
    joints.append(np.array([landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z]))
    joints.append(np.array([landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z]))
    joints.append(np.array([landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z]))
    joints.append(np.array([landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z]))
    joints.append(np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z]))
    joints.append(find_mid([landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z],
                           [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z])
                  )
    joints.append(np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z]))
    joints.append(np.array([landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z]))
    joints.append(np.array([landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z]))
    joints.append(np.array([landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z]))
    joints.append(np.array([landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z]))

    return joints


def get_body_lines(joints):
    body_lines = []
    body_lines.append(get_line(joints[0], joints[1]))  # head to neck
    body_lines.append(get_line(joints[1], joints[9]))  # neck to hip
    body_lines.append(get_line(joints[8], joints[9]))  # left_hip to mid_hip
    body_lines.append(get_line(joints[10], joints[9]))  # right_hip to mid_hip
    body_lines.append(get_line(joints[8], joints[11]))  # left_hip to left_knee
    # right_hip to right_knee
    body_lines.append(get_line(joints[10], joints[12]))
    # left_knee to left_ankle
    body_lines.append(get_line(joints[11], joints[13]))
    # right_knee to right_ankle
    body_lines.append(get_line(joints[12], joints[14]))
    body_lines.append(get_line(joints[2], joints[1]))  # left_shoulder to neck
    body_lines.append(get_line(joints[3], joints[1]))  # right_shoulder to neck
    # left_shoulder to left_elbow
    body_lines.append(get_line(joints[2], joints[4]))
    # left_elbow to left_wrist
    body_lines.append(get_line(joints[4], joints[6]))
    # right_shoulder to right_elbow
    body_lines.append(get_line(joints[3], joints[5]))
    # right_elbow to right_wrist
    body_lines.append(get_line(joints[5], joints[7]))
#     for i in range(0,len(joints)):
#         for j in range(i+1,len(joints)):
#             body_lines.append(get_line(joint[i],joints[j]))
    # print("body_lines")
    # print(len(body_lines))
    # print(body_lines)
    return body_lines


def get_body_angles(body_lines):
    body_angles = []
    for i in range(0, len(body_lines)):
        for j in range(i+1, len(body_lines)):
            # print("i,j ")
            # print(i)
            # print(j)
            numerator = np.dot(body_lines[i], body_lines[j])
            # print("numerator")
            # print(numerator)
            denominator = magnitude(body_lines[i])*magnitude(body_lines[j])
            # print("denominator")
            # print(denominator)
            argument = max(-1.0, min(numerator / denominator, 1.0))
            body_angles.append(math.acos(argument))
            # print("No eerrrorr")
    return body_angles


def calculate_features(image):
    output = []
    # Checking if the image is empty or not
    if image is None:
        result = "Image is empty!!"
        return
    # image=image_resize(image)

    # Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        results = pose.process(image)
        # print("results")
        # print(results)
        if not results.pose_landmarks:
            return
        try:
            landmarks = results.pose_landmarks.landmark
            # print("landmarks")
            # print(landmarks)
            output = []
            joints = get_joints(landmarks)
            # print("joints")
            # print(joints)
#             output=output+joints
            # get_body_lines(joints)
            bodyLines = get_body_lines(joints)
            output = output+get_body_angles(bodyLines)
            # print("output")
            # print(output)
            return output
        except Exception as e:
            print("Error in calculateFetaure :")
            print(e)


def process_single_frame(frame):
    confidence = 0
    max_confidence = 0
    prev_confidence = 0

    # Check if the frame is None
    if frame is None:
        print("Error: Frame is None")
        return {'status': 'Error: Frame is None'}

    # Check if the frame is a valid numpy array
    if not isinstance(frame, np.ndarray):
        print("Error: Invalid frame format")
        return {'status': 'Error: Invalid frame format'}

    # Check if the frame has the correct number of channels (3 for BGR)
    if frame.shape[-1] != 3:
        print("Error: Invalid number of channels in frame")
        return {'status': 'Error: Invalid number of channels in frame'}

    # Setup mediapipe instance
    label_mapping = {0: 'Bhujangasana', 1: 'JanuSirasana',
                     2: 'Padmasana', 3: 'Savasana', 4: 'Tadasana', 5: 'Trikonasana'}

    # Recolor image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Make detection
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        results = pose.process(image)

    # Recolor back to BGR
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Extract landmarks
    try:
        landmarks = results.pose_landmarks.landmark
        test = [] + get_body_angles(get_body_lines(get_joints(landmarks)))
        asana = classifier.predict(np.array(test).reshape(1, -1))
        asana_name = label_mapping.get(asana[0], 'Unknown')
        confidence = sorted(classifier.predict_proba(
            np.array(test).reshape(1, -1))[0])[-1]

        if (confidence >= 0.75 and confidence > prev_confidence):
            max_confidence = confidence
            similar = kNN.kneighbors(X=np.array(test).reshape(
                1, -1), return_distance=True)[1][0][0]
            ideal_features = df.iloc[similar]
            curr_features = test
            matrix = []
            matrix.append(ideal_features.feature_1-curr_features[0])
            matrix.append(ideal_features.feature_78-curr_features[77])
            matrix.append(ideal_features.feature_84-curr_features[83])
            matrix.append(ideal_features.feature_86-curr_features[85])
            matrix.append(ideal_features.feature_91-curr_features[90])
            matrix.append(ideal_features.feature_16-curr_features[15])
            matrix.append(ideal_features.feature_17-curr_features[16])
            matrix.append(ideal_features.feature_48-curr_features[47])
            matrix.append(ideal_features.feature_57-curr_features[56])

            advice = []
            for diff in matrix:
                diff = diff*180/math.pi
                if (abs(diff) <= 13.9):
                    advice.append("Correct")
                elif (diff < 0):
                    advice.append('Open more')
                else:
                    advice.append('Bend more')

            if confidence >= 0.75 and len(np.unique(advice)) == 1 and advice[0] == 'Correct':
                advice_text = {'status': 'Correct pose'}
            elif max_confidence > 0:
                advice_text = {'status': 'Pose advice', 'asana': asana_name, 'confidence': ((confidence*1000000)//100)/100,
                               'left_shoulder': advice[1], 'right_shoulder': advice[2], 'left_elbow': advice[3],
                               'right_elbow': advice[4], 'left_hip': advice[5], 'right_hip': advice[6],
                               'left_knee': advice[7], 'right_knee': advice[8]}
                prev_confidence = confidence

        else:
            advice_text = {'status': 'Pose not confident enough'}

    except Exception as e:
        advice_text = {'status': 'Error in processing'}

    print(advice_text)
    return advice_text


def process(frames):
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(r"check1.mp4")
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    confidence = 0
    max_confidence = 0
    prev_confidence = 0
    results2 = 0
    # Setup mediapipe instance
    # similar_image=None
    label_mapping = {0: 'Bhujangasana', 1: 'JanuSirasana',
                     2: 'Padmasana', 3: 'Savasana', 4: 'Tadasana', 5: 'Trikonasana'}
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            # if similar_image is None:
            #     similar_image=cv2.imread(r'none.jpg', cv2.IMREAD_COLOR)
            #     # similar_image=image_resize_small(similar_image)
            if ret:
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # image=image_resize(image)
                image.flags.writeable = False

                # Make detection
                results = pose.process(image)

                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark
                    test = [] + \
                        get_body_angles(get_body_lines(get_joints(landmarks)))
    #                 print(test)
                    asana = classifier.predict(np.array(test).reshape(1, -1))
                    asana_name = label_mapping.get(asana[0], 'Unknown')
                    confidence = sorted(classifier.predict_proba(
                        np.array(test).reshape(1, -1))[0])[-1]
                    cv2.putText(image, 'Asana = '+asana_name,
                                (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,
                                                              0, 255), 2, cv2.LINE_AA
                                )
                    cv2.putText(image, 'Confidence = '+str(((confidence*1000000)//100)/100)+'%',
                                (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,
                                                              0, 255), 2, cv2.LINE_AA
                                )
                    if (confidence >= 0.75 and confidence > prev_confidence):
                        max_confidence = confidence
                        similar = kNN.kneighbors(X=np.array(test).reshape(
                            1, -1), return_distance=True)[1][0][0]
                        ideal_features = df.iloc[similar]
                        curr_features = test
                        matrix = []
                        matrix.append(ideal_features.feature_1 -
                                      curr_features[0])
                        matrix.append(
                            ideal_features.feature_78-curr_features[77])
                        matrix.append(
                            ideal_features.feature_84-curr_features[83])
                        matrix.append(
                            ideal_features.feature_86-curr_features[85])
                        matrix.append(
                            ideal_features.feature_91-curr_features[90])
                        matrix.append(
                            ideal_features.feature_16-curr_features[15])
                        matrix.append(
                            ideal_features.feature_17-curr_features[16])
                        matrix.append(
                            ideal_features.feature_48-curr_features[47])
                        matrix.append(
                            ideal_features.feature_57-curr_features[56])
    #                     matrix=matrix/math.pi
                        advice = []
                        for diff in matrix:
                            diff = diff*180/math.pi
    #                         advice.append(str(diff))
                            if (abs(diff) <= 13.9):
                                advice.append("Correct")
                            elif (diff < 0):
                                advice.append('Open more')
                            else:
                                advice.append('Bend more')
    #                         else:
    #                             advice.append(str(diff))

    #                     print(features)

                        asana = asana_names[similar]
                        # file_name=file_names[similar]
                        # similar_image=cv2.imread(os.path.join('Images', asana, file_name), cv2.IMREAD_COLOR)
                        # similar_image=image_resize_small(similar_image)

                    if confidence >= 0.75 and len(np.unique(advice)) == 1 and advice[0] == 'Correct':
                        cv2.putText(image, 'Correct pose',
                                    (470, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (
                                        0, 255, 0), 2, cv2.LINE_AA
                                    )
                    elif max_confidence > 0:
                        #                     cv2.putText(image, 'Neck = '+advice[0],
                        #                            (510, 50),
                        #                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 1, cv2.LINE_AA
                        #                                 )
                        cv2.putText(image, 'left_shoulder= '+advice[1],
                                    (800, 190),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (
                                        255, 0, 0), 1, cv2.LINE_AA
                                    )
                        cv2.putText(image, 'right_shoulder = '+advice[2],
                                    (100, 190),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (
                                        255, 0, 0), 1, cv2.LINE_AA
                                    )
                        cv2.putText(image, 'left_elbow = '+advice[3],
                                    (820, 300),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (
                                        255, 0, 0), 1, cv2.LINE_AA
                                    )
                        cv2.putText(image, 'right_elbow = '+advice[4],
                                    (90, 300),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (
                                        255, 0, 0), 1, cv2.LINE_AA
                                    )
                        cv2.putText(image, 'left_hip = '+advice[5],
                                    (800, 400),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (
                                        255, 0, 0), 1, cv2.LINE_AA
                                    )
                        cv2.putText(image, 'right_hip = '+advice[6],
                                    (100, 400),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (
                                        255, 0, 0), 1, cv2.LINE_AA
                                    )
                        cv2.putText(image, 'left_knee = '+advice[7],
                                    (800, 500),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (
                                        255, 0, 0), 1, cv2.LINE_AA
                                    )
                        cv2.putText(image, 'right_knee = '+advice[8],
                                    (100, 500),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (
                                        255, 0, 0), 1, cv2.LINE_AA
                                    )
                        prev_confidence = confidence

                except Exception as e:
                    #                 print(e)
                    pass

                # Render detections
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                cv2.imshow('Mediapipe Feed', image)
                # cv2.imshow('Similar_image', similar_image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            else:
                break

        cap.release()
        cv2.destroyAllWindows()


# @app.route('/analyze-pose', methods=['POST'])
def analyze_pose():
    try:
        data = request.get_json()
        image_data = data['image'].split(',')[1]
        img_np = np.frombuffer(base64.b64decode(image_data), dtype=np.uint8)
        image = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        result = process_pose(image)

        if result:
            return jsonify(result)
        else:
            return jsonify({'error': 'Failed to process pose'}), 500

    except Exception as e:
        print("Error in analyze_pose:", e)
        return jsonify({'error': 'Internal Server Error'}), 500


def _build_cors_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add('Access-Control-Allow-Headers', "*")
    response.headers.add('Access-Control-Allow-Methods', "*")
    return response


def _corsify_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


if __name__ == '__main__':
    socketio.run(app, debug=True)
