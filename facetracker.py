import copy
import os
import sys
import argparse
import traceback
import gc
import mediapipe as mp
mp_hands = mp.solutions.hands

# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--ip", help="Set IP address for sending tracking data", default="127.0.0.1")
parser.add_argument("-p", "--port", type=int, help="Set port for sending tracking data", default=11573)
if os.name == 'nt':
    parser.add_argument("-l", "--list-cameras", type=int, help="Set this to 1 to list the available cameras and quit, set this to 2 or higher to output only the names", default=0)
    parser.add_argument("-a", "--list-dcaps", type=int, help="Set this to -1 to list all cameras and their available capabilities, set this to a camera id to list that camera's capabilities", default=None)
    parser.add_argument("-W", "--width", type=int, help="Set camera and raw RGB width", default=640)
    parser.add_argument("-H", "--height", type=int, help="Set camera and raw RGB height", default=360)
    parser.add_argument("-D", "--dcap", type=int, help="Set which device capability line to use or -1 to use the default camera settings (FPS still need to be set separately)", default=None)
    parser.add_argument("-B", "--blackmagic", type=int, help="When set to 1, special support for Blackmagic devices is enabled", default=0)
else:
    parser.add_argument("-W", "--width", type=int, help="Set raw RGB width", default=640)
    parser.add_argument("-H", "--height", type=int, help="Set raw RGB height", default=360)
parser.add_argument("-F", "--fps", type=int, help="Set camera frames per second", default=24)
parser.add_argument("-c", "--capture", help="Set camera ID (0, 1...) or video file", default="0")
parser.add_argument("-M", "--mirror-input", action="store_true", help="Process a mirror image of the input video")
parser.add_argument("-m", "--max-threads", type=int, help="Set the maximum number of threads", default=1)
parser.add_argument("-t", "--threshold", type=float, help="Set minimum confidence threshold for face tracking", default=None)
parser.add_argument("-d", "--detection-threshold", type=float, help="Set minimum confidence threshold for face detection", default=0.6)
parser.add_argument("-v", "--visualize", type=int, help="Set this to 1 to visualize the tracking, to 2 to also show face ids, to 3 to add confidence values or to 4 to add numbers to the point display", default=0)
parser.add_argument("-P", "--pnp-points", type=int, help="Set this to 1 to add the 3D fitting points to the visualization", default=0)
parser.add_argument("-s", "--silent", type=int, help="Set this to 1 to prevent text output on the console", default=0)
parser.add_argument("--hands", type=int, help="Set this to 1 to enable hands tracking, 0 to disable hands tracking", default=0)
parser.add_argument("--fingers", type=int, help="Set this to 1 to enable finger tracking, 0 to disable finger tracking", default=0)
parser.add_argument("--faces", type=int, help="Set the maximum number of faces (slow)", default=1)
parser.add_argument("--scan-retinaface", type=int, help="When set to 1, scanning for additional faces will be performed using RetinaFace in a background thread, otherwise a simpler, faster face detection mechanism is used. When the maximum number of faces is 1, this option does nothing.", default=0)
parser.add_argument("--scan-every", type=int, help="Set after how many frames a scan for new faces should run", default=3)
parser.add_argument("--discard-after", type=int, help="Set the how long the tracker should keep looking for lost faces", default=10)
parser.add_argument("--max-feature-updates", type=int, help="This is the number of seconds after which feature min/max/medium values will no longer be updated once a face has been detected.", default=900)
parser.add_argument("--no-3d-adapt", type=int, help="When set to 1, the 3D face model will not be adapted to increase the fit", default=1)
parser.add_argument("--try-hard", type=int, help="When set to 1, the tracker will try harder to find a face", default=0)
parser.add_argument("--video-out", help="Set this to the filename of an AVI file to save the tracking visualization as a video", default=None)
parser.add_argument("--video-scale", type=int, help="This is a resolution scale factor applied to the saved AVI file", default=1, choices=[1,2,3,4])
parser.add_argument("--video-fps", type=float, help="This sets the frame rate of the output AVI file", default=24)
parser.add_argument("--raw-rgb", type=int, help="When this is set, raw RGB frames of the size given with \"-W\" and \"-H\" are read from standard input instead of reading a video", default=0)
parser.add_argument("--log-data", help="You can set a filename to which tracking data will be logged here", default="")
parser.add_argument("--log-output", help="You can set a filename to console output will be logged here", default="")
parser.add_argument("--model", type=int, help="This can be used to select the tracking model. Higher numbers are models with better tracking quality, but slower speed, except for model 4, which is wink optimized. Models 1 and 0 tend to be too rigid for expression and blink detection. Model -2 is roughly equivalent to model 1, but faster. Model -3 is between models 0 and -1.", default=3, choices=[-3, -2, -1, 0, 1, 2, 3, 4])
parser.add_argument("--model-dir", help="This can be used to specify the path to the directory containing the .onnx model files", default=None)
parser.add_argument("--gaze-tracking", type=int, help="When set to 1, gaze tracking is enabled, which makes things slightly slower", default=1)
parser.add_argument("--face-id-offset", type=int, help="When set, this offset is added to all face ids, which can be useful for mixing tracking data from multiple network sources", default=0)
parser.add_argument("--repeat-video", type=int, help="When set to 1 and a video file was specified with -c, the tracker will loop the video until interrupted", default=0)
parser.add_argument("--dump-points", type=str, help="When set to a filename, the current face 3D points are made symmetric and dumped to the given file when quitting the visualization with the \"q\" key", default="")
parser.add_argument("--benchmark", type=int, help="When set to 1, the different tracking models are benchmarked, starting with the best and ending with the fastest and with gaze tracking disabled for models with negative IDs", default=0)
parser.add_argument("--frame-data", type=int, help="When set to 1, the server is sending webcam frame data", default=0)
parser.add_argument("--calibrate", type=int, help="When set to 1, calibration data will be written to calibrate-output", default=0)
parser.add_argument("--calibration-path", help="Path to calibration data save file", default=None)



if os.name == 'nt':
    parser.add_argument("--use-dshowcapture", type=int, help="When set to 1, libdshowcapture will be used for video input instead of OpenCV", default=1)
    parser.add_argument("--blackmagic-options", type=str, help="When set, this additional option string is passed to the blackmagic capture library", default=None)
    parser.add_argument("--priority", type=int, help="When set, the process priority will be changed", default=None, choices=[0, 1, 2, 3, 4, 5])
args = parser.parse_args()

os.environ["OMP_NUM_THREADS"] = str(args.max_threads)

class OutputLog(object):
    def __init__(self, fh, output):
        self.fh = fh
        self.output = output
    def write(self, buf):
        if self.fh is not None:
            self.fh.write(buf)
        self.output.write(buf)
        self.flush()
    def flush(self):
        if self.fh is not None:
            self.fh.flush()
        self.output.flush()
output_logfile = None
if args.log_output != "":
    output_logfile = open(args.log_output, "w")
sys.stdout = OutputLog(output_logfile, sys.stdout)
sys.stderr = OutputLog(output_logfile, sys.stderr)

if os.name == 'nt':
    import dshowcapture
    if args.blackmagic == 1:
        dshowcapture.set_bm_enabled(True)
    if args.blackmagic_options is not None:
        dshowcapture.set_options(args.blackmagic_options)
    if args.priority is not None:
        import psutil
        classes = [psutil.IDLE_PRIORITY_CLASS, psutil.BELOW_NORMAL_PRIORITY_CLASS, psutil.NORMAL_PRIORITY_CLASS, psutil.ABOVE_NORMAL_PRIORITY_CLASS, psutil.HIGH_PRIORITY_CLASS, psutil.REALTIME_PRIORITY_CLASS]
        p = psutil.Process(os.getpid())
        p.nice(classes[args.priority])

if os.name == 'nt' and (args.list_cameras > 0 or args.list_dcaps is not None):
    cap = dshowcapture.DShowCapture()
    info = cap.get_info()
    unit = 10000000.;
    if args.list_dcaps is not None:
        formats = {0: "Any", 1: "Unknown", 100: "ARGB", 101: "XRGB", 200: "I420", 201: "NV12", 202: "YV12", 203: "Y800", 300: "YVYU", 301: "YUY2", 302: "UYVY", 303: "HDYC (Unsupported)", 400: "MJPEG", 401: "H264" }
        for cam in info:
            if args.list_dcaps == -1:
                type = ""
                if cam['type'] == "Blackmagic":
                    type = "Blackmagic: "
                print(f"{cam['index']}: {type}{cam['name']}")
            if args.list_dcaps != -1 and args.list_dcaps != cam['index']:
                continue
            for caps in cam['caps']:
                format = caps['format']
                if caps['format'] in formats:
                    format = formats[caps['format']]
                if caps['minCX'] == caps['maxCX'] and caps['minCY'] == caps['maxCY']:
                    print(f"    {caps['id']}: Resolution: {caps['minCX']}x{caps['minCY']} FPS: {unit/caps['maxInterval']:.3f}-{unit/caps['minInterval']:.3f} Format: {format}")
                else:
                    print(f"    {caps['id']}: Resolution: {caps['minCX']}x{caps['minCY']}-{caps['maxCX']}x{caps['maxCY']} FPS: {unit/caps['maxInterval']:.3f}-{unit/caps['minInterval']:.3f} Format: {format}")
    else:
        if args.list_cameras == 1:
            print("Available cameras:")
        for cam in info:
            type = ""
            if cam['type'] == "Blackmagic":
                type = "Blackmagic: "
            if args.list_cameras == 1:
                print(f"{cam['index']}: {type}{cam['name']}")
            else:
                print(f"{type}{cam['name']}")
    cap.destroy_capture()
    sys.exit(0)

import math
import numpy as np
import time
import cv2
import socket
import struct
import json
from input_reader import InputReader, VideoReader, DShowCaptureReader, try_int
from tracker import Tracker, get_model_base_path
import pickle

def saveCalibrationData():
    if args.calibrate == 1 and args.calibration_path != None and len(tracker.face_info) > 0:
        db = tracker.face_info[0].features
        with open(args.calibration_path, 'ab') as dbfile: 
            dbfile = open(args.calibration_path, 'ab')
            # source, destination
            pickle.dump(db, dbfile)  

if args.benchmark > 0:
    model_base_path = get_model_base_path(args.model_dir)
    im = cv2.imread(os.path.join(model_base_path, "benchmark.bin"), cv2.IMREAD_COLOR)
    results = []
    for model_type in [3, 2, 1, 0, -1, -2, -3]:
        tracker = Tracker(224, 224, threshold=0.1, max_threads=args.max_threads, max_faces=1, discard_after=0, scan_every=0, silent=True, model_type=model_type, model_dir=args.model_dir, no_gaze=(model_type == -1), detection_threshold=0.1, use_retinaface=0, max_feature_updates=900, static_model=True if args.no_3d_adapt == 1 else False)
        tracker.detected = 1
        tracker.faces = [(0, 0, 224, 224)]
        total = 0.0
        for i in range(100):
            start = time.perf_counter()
            r = tracker.predict(im)
            total += time.perf_counter() - start
        print(1. / (total / 100.))
    sys.exit(0)

max_length = 65535-28 # 28 is for UDP header


target_ip = args.ip
target_port = args.port

if args.faces >= 40:
    print("Transmission of tracking data over network is not supported with 40 or more faces.")

fps = args.fps
dcap = None
use_dshowcapture_flag = False
if os.name == 'nt':
    dcap = args.dcap
    use_dshowcapture_flag = True if args.use_dshowcapture == 1 else False
    input_reader = InputReader(args.capture, args.raw_rgb, args.width, args.height, fps, use_dshowcapture=use_dshowcapture_flag, dcap=dcap)
    if args.dcap == -1 and type(input_reader) == DShowCaptureReader:
        fps = min(fps, input_reader.device.get_fps())
else:
    input_reader = InputReader(args.capture, args.raw_rgb, args.width, args.height, fps)
if type(input_reader.reader) == VideoReader:
    fps = 0

log = None
out = None
first = True
height = 0
width = 0
tracker = None
sock = None
total_tracking_time = 0.0
tracking_time = 0.0
tracking_frames = 0
frame_count = 0

features = ["eye_l", "eye_r", "eyebrow_steepness_l", "eyebrow_updown_l", "eyebrow_quirk_l", "eyebrow_steepness_r", "eyebrow_updown_r", "eyebrow_quirk_r", "mouth_corner_updown_l", "mouth_corner_inout_l", "mouth_corner_updown_r", "mouth_corner_inout_r", "mouth_open", "mouth_wide"]

if args.log_data != "":
    log = open(args.log_data, "w")
    log.write("Frame,Time,Width,Height,FPS,Face,FaceID,RightOpen,LeftOpen,AverageConfidence,Success3D,PnPError,RotationQuat.X,RotationQuat.Y,RotationQuat.Z,RotationQuat.W,Euler.X,Euler.Y,Euler.Z,RVec.X,RVec.Y,RVec.Z,TVec.X,TVec.Y,TVec.Z")
    for i in range(66):
        log.write(f",Landmark[{i}].X,Landmark[{i}].Y,Landmark[{i}].Confidence")
    for i in range(66):
        log.write(f",Point3D[{i}].X,Point3D[{i}].Y,Point3D[{i}].Z")
    for feature in features:
        log.write(f",{feature}")
    log.write("\r\n")
    log.flush()

is_camera = args.capture == str(try_int(args.capture))

try:
    attempt = 0
    frame_time = time.perf_counter()
    target_duration = 0
    if fps > 0:
        target_duration = 1. / float(fps)
    repeat = args.repeat_video != 0 and type(input_reader.reader) == VideoReader
    need_reinit = 0
    failures = 0
    source_name = input_reader.name
    if args.hands == 1:
        holistic = mp_hands.Hands(model_complexity=1,min_detection_confidence=0.82,min_tracking_confidence=0.82)
        
    while repeat or input_reader.is_open(): 
        if not input_reader.is_open() or need_reinit == 1:
            input_reader = InputReader(args.capture, args.raw_rgb, args.width, args.height, fps, use_dshowcapture=use_dshowcapture_flag, dcap=dcap)
            if input_reader.name != source_name:
                print(f"Failed to reinitialize camera and got {input_reader.name} instead of {source_name}.")
                sys.exit(1)
            need_reinit = 2
            time.sleep(0.02)
            continue
        if not input_reader.is_ready():
            time.sleep(0.02)
            continue

        ret, frame = input_reader.read()
        if ret and args.mirror_input:
            frame = cv2.flip(frame, 1)
        if not ret:
            if repeat:
                if need_reinit == 0:
                    need_reinit = 1
                continue
            elif is_camera:
                attempt += 1
                if attempt > 30:
                    break
                else:
                    time.sleep(0.02)
                    if attempt == 3:
                        need_reinit = 1
                    continue
            else:
                break;

        attempt = 0
        need_reinit = 0
        frame_count += 1
        now = time.time()
        
        if first:
            first = False
            height, width, channels = frame.shape
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            calibration = -1 if args.calibration_path != None else args.max_feature_updates
            tracker = Tracker(width, height, threshold=args.threshold, max_threads=args.max_threads, max_faces=args.faces, discard_after=args.discard_after, scan_every=args.scan_every, silent=False if args.silent == 0 else True, model_type=args.model, model_dir=args.model_dir, no_gaze=False if args.gaze_tracking != 0 and args.model != -1 else True, detection_threshold=args.detection_threshold, use_retinaface=args.scan_retinaface, max_feature_updates=calibration, static_model=True if args.no_3d_adapt == 1 else False, try_hard=args.try_hard == 1)
            if args.video_out is not None:
                out = cv2.VideoWriter(args.video_out, cv2.VideoWriter_fourcc('F','F','V','1'), args.video_fps, (width * args.video_scale, height * args.video_scale))
            if args.calibrate != 1 and args.calibration_path != None and os.path.isfile(args.calibration_path): #only load values if not in recording mode
                with open(args.calibration_path, 'rb') as dbfile: 
                    db = pickle.load(dbfile)
                    tracker.face_info[0].features = db
                    tracker.first_seen = -1
                    tracker.updating = False

        try:
            inference_start = time.perf_counter()
            faces = tracker.predict(frame)
            if len(faces) > 0:
                inference_time = (time.perf_counter() - inference_start)
                total_tracking_time += inference_time
                tracking_time += inference_time / len(faces)
                tracking_frames += 1
            packet = bytearray()
            detected = False
            for face_num, f in enumerate(faces):
                f = copy.copy(f)
                f.id += args.face_id_offset
                if f.eye_blink is None:
                    f.eye_blink = [1, 1]
                right_state = "O" if f.eye_blink[0] > 0.30 else "-"
                left_state = "O" if f.eye_blink[1] > 0.30 else "-"
                if args.silent == 0:
                    print(f"Confidence[{f.id}]: {f.conf:.4f} / 3D fitting error: {f.pnp_error:.4f} / Eyes: {left_state}, {right_state}")
                detected = True
                if not f.success:
                    pts_3d = np.zeros((70, 3), np.float32)
                packet.extend(bytearray(struct.pack("d", now)))
                packet.extend(bytearray(struct.pack("i", f.id)))
                packet.extend(bytearray(struct.pack("f", width)))
                packet.extend(bytearray(struct.pack("f", height)))
                packet.extend(bytearray(struct.pack("f", f.eye_blink[0])))
                packet.extend(bytearray(struct.pack("f", f.eye_blink[1])))
                packet.extend(bytearray(struct.pack("B", 1 if f.success else 0)))
                packet.extend(bytearray(struct.pack("f", f.pnp_error)))
                packet.extend(bytearray(struct.pack("f", f.quaternion[0])))
                packet.extend(bytearray(struct.pack("f", f.quaternion[1])))
                packet.extend(bytearray(struct.pack("f", f.quaternion[2])))
                packet.extend(bytearray(struct.pack("f", f.quaternion[3])))
                packet.extend(bytearray(struct.pack("f", f.euler[0])))
                packet.extend(bytearray(struct.pack("f", f.euler[1])))
                packet.extend(bytearray(struct.pack("f", f.euler[2])))
                packet.extend(bytearray(struct.pack("f", f.translation[0])))
                packet.extend(bytearray(struct.pack("f", f.translation[1])))
                packet.extend(bytearray(struct.pack("f", f.translation[2])))
                if log is not None:
                    log.write(f"{frame_count},{now},{width},{height},{fps},{face_num},{f.id},{f.eye_blink[0]},{f.eye_blink[1]},{f.conf},{f.success},{f.pnp_error},{f.quaternion[0]},{f.quaternion[1]},{f.quaternion[2]},{f.quaternion[3]},{f.euler[0]},{f.euler[1]},{f.euler[2]},{f.rotation[0]},{f.rotation[1]},{f.rotation[2]},{f.translation[0]},{f.translation[1]},{f.translation[2]}")
                for (x,y,c) in f.lms:
                    packet.extend(bytearray(struct.pack("f", c)))
                if args.visualize > 1:
                    frame = cv2.putText(frame, str(f.id), (int(f.bbox[0]), int(f.bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,255))
                if args.visualize > 2:
                    frame = cv2.putText(frame, f"{f.conf:.4f}", (int(f.bbox[0] + 18), int(f.bbox[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))
                for pt_num, (x,y,c) in enumerate(f.lms):
                    packet.extend(bytearray(struct.pack("f", y)))
                    packet.extend(bytearray(struct.pack("f", x)))
                    if log is not None:
                        log.write(f",{y},{x},{c}")
                    if pt_num == 66 and (f.eye_blink[0] < 0.30 or c < 0.20):
                        continue
                    if pt_num == 67 and (f.eye_blink[1] < 0.30 or c < 0.20):
                        continue
                    x = int(x + 0.5)
                    y = int(y + 0.5)
                    if args.visualize != 0 or out is not None:
                        if args.visualize > 3:
                            frame = cv2.putText(frame, str(pt_num), (int(y), int(x)), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255,255,0))
                        color = (0, 255, 0)
                        dot_size = int(width/500)
                        if pt_num >= 66:
                            color = (255, 255, 0)
                        if not (x < 0 or y < 0 or x >= height or y >= width):
                            cv2.circle(frame, (y, x), dot_size, color, -1)
                if args.pnp_points != 0 and (args.visualize != 0 or out is not None) and f.rotation is not None:
                    if args.pnp_points > 1:
                        projected = cv2.projectPoints(f.face_3d[0:66], f.rotation, f.translation, tracker.camera, tracker.dist_coeffs)
                    else:
                        projected = cv2.projectPoints(f.contour, f.rotation, f.translation, tracker.camera, tracker.dist_coeffs)
                    for [(x,y)] in projected[0]:
                        x = int(x + 0.5)
                        y = int(y + 0.5)
                        if not (x < 0 or y < 0 or x >= height or y >= width):
                            frame[int(x), int(y)] = (0, 255, 255)
                        x += 1
                        if not (x < 0 or y < 0 or x >= height or y >= width):
                            frame[int(x), int(y)] = (0, 255, 255)
                        y += 1
                        if not (x < 0 or y < 0 or x >= height or y >= width):
                            frame[int(x), int(y)] = (0, 255, 255)
                        x -= 1
                        if not (x < 0 or y < 0 or x >= height or y >= width):
                            frame[int(x), int(y)] = (0, 255, 255)
                for (x,y,z) in f.pts_3d:
                    packet.extend(bytearray(struct.pack("f", x)))
                    packet.extend(bytearray(struct.pack("f", -y)))
                    packet.extend(bytearray(struct.pack("f", -z)))
                    if log is not None:
                        log.write(f",{x},{-y},{-z}")
                if f.current_features is None:
                    f.current_features = {}
                for feature in features:
                    if not feature in f.current_features:
                        f.current_features[feature] = 0
                    packet.extend(bytearray(struct.pack("f", f.current_features[feature])))
                    if log is not None:
                        log.write(f",{f.current_features[feature]}")
                if log is not None:
                    log.write("\r\n")
                    log.flush()

            handDetected = False

            # -- DETECT HANDS -----------------------------
            if args.hands == 1:
                frame.flags.writeable = False
                # frame = cv2.flip(frame, 1)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(frame)

                frame.flags.writeable = True
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                # ---------------------------------------------
                
                # hands_count = 0 if not results.multi_handedness else len(results.multi_handedness)
                # packet.extend(bytearray(struct.pack("i", hands_count)))
                if args.fingers == 1:
                    handposition = [[[0] * 3 for i in range(25)] for i in range(2)] # 4+21 vector3 for 2 hands
                else:
                    handposition = [[[0] * 3 for i in range(4)] for i in range(2)] # 4 vector3 for 2 hands
                    
                if results.multi_handedness:                    
                    for i, handedness in enumerate(results.multi_handedness):
                        hand_landmarks = results.multi_hand_landmarks[i]
                        jointIndices = [0,1,5,9]
                        poses = [hand_landmarks.landmark[i] for i in jointIndices] # wrist, index, middle joints 
                        
                        # draw a circle on a hand
                        wrist = poses[0]
                        middle = poses[3]
                        x = (middle.x + wrist.x)/2 * width
                        y = (middle.y + wrist.y)/2.2 * height
                        frame = cv2.circle(frame,(int(x), int(y)), int(width/50), (0,255,0), 2)
                        
                        if middle != wrist:
                            handDetected = True
                        
                        # Add fingers world coordinate info if needed
                        if args.fingers == 1:
                            poses = [*poses, *results.multi_hand_world_landmarks[i].landmark]
                        
                        hand_info  = handedness.classification[0]
                        hand_index = hand_info.index

                        for jointIndex, pose in enumerate(poses):
                            handposition[hand_index][jointIndex][0] = float(pose.x)                    
                            handposition[hand_index][jointIndex][1] = float(pose.y)                
                            handposition[hand_index][jointIndex][2] = float(pose.z)
                
                
                if handDetected:
                    if not detected:
                        packet.extend(bytearray(struct.pack("d", now)))
                        packet.extend(bytearray(struct.pack("i", -1)))
                    for hand in handposition:
                        for joint in hand:
                            for axis in joint:
                                packet.extend(bytearray(struct.pack("f", axis)))

            if (detected and len(faces) < 40) or handDetected:
                sock.sendto(packet, (target_ip, target_port))         
            else:
                sock.sendto(struct.pack("B", 0), (target_ip, target_port))


            if args.frame_data == 1:
                cam_frame = frame if width <= 480 else cv2.resize(frame, (480, math.ceil(height * (480 / width))), interpolation=cv2.INTER_NEAREST)
                retval, buffer = cv2.imencode(".jpg", cam_frame)
                if retval:
                    # convert to byte array
                    buffer = buffer.tobytes()
                    # get size of the frame
                    buffer_size = len(buffer)

                    num_of_packs = 1
                    if buffer_size > max_length:
                        num_of_packs = math.ceil(buffer_size/max_length)

                    # frame_info = {"packs":num_of_packs}
                    frame_info = bytearray(struct.pack("i", num_of_packs))

                    # send the number of packs to be expected
                    # print("Number of packs:", num_of_packs)
                    sock.sendto(frame_info, (target_ip, target_port))
                    
                    left = 0
                    right = max_length

                    for i in range(num_of_packs):
                        # print("left:", left)
                        # print("right:", right)

                        # truncate data to send
                        data = buffer[left:right]
                        left = right
                        right += max_length

                        # send the frames accordingly
                        sock.sendto(data, (target_ip, target_port))

            if out is not None:
                video_frame = frame
                if args.video_scale != 1:
                    video_frame = cv2.resize(frame, (width * args.video_scale, height * args.video_scale), interpolation=cv2.INTER_NEAREST)
                out.write(video_frame)
                if args.video_scale != 1:
                    del video_frame


            if args.visualize != 0 and args.frame_data == 0:
                cv2.imshow('OpenSeeFace Visualization', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    if args.dump_points != "" and faces is not None and len(faces) > 0:
                        np.set_printoptions(threshold=sys.maxsize, precision=15)
                        pairs = [
                            (0, 16),
                            (1, 15),
                            (2, 14),
                            (3, 13),
                            (4, 12),
                            (5, 11),
                            (6, 10),
                            (7, 9),
                            (17, 26),
                            (18, 25),
                            (19, 24),
                            (20, 23),
                            (21, 22),
                            (31, 35),
                            (32, 34),
                            (36, 45),
                            (37, 44),
                            (38, 43),
                            (39, 42),
                            (40, 47),
                            (41, 46),
                            (48, 52),
                            (49, 51),
                            (56, 54),
                            (57, 53),
                            (58, 62),
                            (59, 61),
                            (65, 63)
                        ]
                        points = copy.copy(faces[0].face_3d)
                        for a, b in pairs:
                            x = (points[a, 0] - points[b, 0]) / 2.0
                            y = (points[a, 1] + points[b, 1]) / 2.0
                            z = (points[a, 2] + points[b, 2]) / 2.0
                            points[a, 0] = x
                            points[b, 0] = -x
                            points[[a, b], 1] = y
                            points[[a, b], 2] = z
                        points[[8, 27, 28, 29, 33, 50, 55, 60, 64], 0] = 0.0
                        points[30, :] = 0.0
                        with open(args.dump_points, "w") as fh:
                            fh.write(repr(points))
                    break
            failures = 0
        except Exception as e:
            if e.__class__ == KeyboardInterrupt:
                if args.silent == 0:
                    print("Quitting")
                break
            traceback.print_exc()
            failures += 1
            if failures > 30:
                break

        collected = False
        del frame
        
        if args.calibrate == 1 and frame_count % 10:
            saveCalibrationData()

        duration = time.perf_counter() - frame_time
        while duration < target_duration:
            if not collected:
                gc.collect()
                collected = True
            duration = time.perf_counter() - frame_time
            sleep_time = target_duration - duration
            if sleep_time > 0:
                time.sleep(sleep_time)
            duration = time.perf_counter() - frame_time
        frame_time = time.perf_counter()
except KeyboardInterrupt:
    if args.silent == 0:
        print("Quitting")
        
if args.hands == 1 and holistic is not None:
    holistic.close()
input_reader.close()
if out is not None:
    out.release()
cv2.destroyAllWindows()

if args.silent == 0 and tracking_frames > 0:
    average_tracking_time = 1000 * tracking_time / tracking_frames
    print(f"Average tracking time per detected face: {average_tracking_time:.2f} ms")
    print(f"Tracking time: {total_tracking_time:.3f} s\nFrames: {tracking_frames}")
