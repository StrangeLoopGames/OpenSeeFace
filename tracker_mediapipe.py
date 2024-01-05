import os
import numpy as np
import math
import cv2
import time
import queue
import threading
import copy
from similaritytransform import SimilarityTransform
from retinaface import RetinaFaceDetector
from remedian import remedian

import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_holistic = mp.solutions.holistic

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def resolve(name):
    f = os.path.join(os.path.dirname(__file__), name)
    return f

def clamp_to_im(pt, w, h):
    x = pt[0]
    y = pt[1]
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if x >= w:
        x = w-1
    if y >= h:
        y = h-1
    return (int(x), int(y+1))

def rotate(origin, point, a):
    a = -a
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(a) * (px - ox) - math.sin(a) * (py - oy)
    qy = oy + math.sin(a) * (px - ox) + math.cos(a) * (py - oy)
    return qx, qy

def angle(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    a = np.arctan2(*(p2 - p1)[::-1])
    return (a % (2 * np.pi))

def compensate(p1, p2):
    a = angle(p1, p2)
    return rotate(p1, p2, a), a

def rotate_image(image, a, center):
    (h, w) = image.shape[:2]
    a = np.rad2deg(a)
    M = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), a, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def intersects(r1, r2, amount=0.3):
    area1 = r1[2] * r1[3]
    area2 = r2[2] * r2[3]
    inter = 0.0
    total = area1 + area2
    
    r1_x1, r1_y1, w, h = r1
    r1_x2 = r1_x1 + w
    r1_y2 = r1_y1 + h
    r2_x1, r2_y1, w, h = r2
    r2_x2 = r2_x1 + w
    r2_y2 = r2_y1 + h

    left = max(r1_x1, r2_x1)
    right = min(r1_x2, r2_x2)
    top = max(r1_y1, r2_y1)
    bottom = min(r1_y2, r2_y2)
    if left < right and top < bottom:
        inter = (right - left) * (bottom - top)
        total -= inter

    if inter / total >= amount:
        return True

    return False

    #return not (r1_x1 > r2_x2 or r1_x2 < r2_x1 or r1_y1 > r2_y2 or r1_y2 < r2_y1)

def group_rects(rects):
    rect_groups = {}
    for rect in rects:
        rect_groups[str(rect)] = [-1, -1, []]
    group_id = 0
    for i, rect in enumerate(rects):
        name = str(rect)
        group = group_id
        group_id += 1
        if rect_groups[name][0] < 0:
            rect_groups[name] = [group, -1, []]
        else:
            group = rect_groups[name][0]
        for j, other_rect in enumerate(rects):
            if i == j:
                continue;
            inter = intersects(rect, other_rect)
            if intersects(rect, other_rect):
                rect_groups[str(other_rect)] = [group, -1, []]
    return rect_groups

def logit(p, factor=16.0):
    if p >= 1.0:
        p = 0.9999999
    if p <= 0.0:
        p = 0.0000001
    p = p/(1-p)
    return float(np.log(p)) / float(factor)

def logit_arr(p, factor=16.0):
    p = np.clip(p, 0.0000001, 0.9999999)
    return np.log(p / (1 - p)) / float(factor)

def matrix_to_quaternion(m):
    t = 0.0
    q = [0.0, 0.0, 0, 0.0]
    if m[2,2] < 0:
        if m[0,0] > m[1,1]:
            t = 1 + m[0,0] - m[1,1] - m[2,2]
            q = [t, m[0,1]+m[1,0], m[2,0]+m[0,2], m[1,2]-m[2,1]]
        else:
            t = 1 - m[0,0] + m[1,1] - m[2,2]
            q = [m[0,1]+m[1,0], t, m[1,2]+m[2,1], m[2,0]-m[0,2]]
    else:
        if m[0,0] < -m[1,1]:
            t = 1 - m[0,0] - m[1,1] + m[2,2]
            q = [m[2,0]+m[0,2], m[1,2]+m[2,1], t, m[0,1]-m[1,0]]
        else:
            t = 1 + m[0,0] + m[1,1] + m[2,2]
            q = [m[1,2]-m[2,1], m[2,0]-m[0,2], m[0,1]-m[1,0], t]
    q = np.array(q, np.float32) * 0.5 / np.sqrt(t)
    return q

def worker_thread(session, frame, input, crop_info, queue, input_name, idx, tracker):
    output = session.run([], {input_name: input})[0]
    conf, lms = tracker.landmarks(output[0], crop_info)
    if conf > tracker.threshold:
        try:
            eye_state = tracker.get_eye_state(frame, lms)
        except:
            eye_state = [(1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)]
        queue.put((session, conf, (lms, eye_state), crop_info, idx))
    else:
        queue.put((session,))

class Feature():
    def __init__(self, threshold=0.15, alpha=0.2, hard_factor=0.15, decay=0.001, max_feature_updates=0):
        self.median = remedian()
        self.min = None
        self.max = None
        self.hard_min = None
        self.hard_max = None
        self.threshold = threshold
        self.alpha = alpha
        self.hard_factor = hard_factor
        self.decay = decay
        self.last = 0
        self.current_median = 0
        self.update_count = 0
        self.max_feature_updates = max_feature_updates
        self.first_seen = -1
        self.updating = True

    def update(self, x, now=0):
        if self.max_feature_updates > 0:
            if self.first_seen == -1:
                self.first_seen = now;
        new = self.update_state(x, now=now)
        filtered = self.last * self.alpha + new * (1 - self.alpha)
        self.last = filtered
        return filtered

    def update_state(self, x, now=0):
        updating = self.updating and (self.max_feature_updates == 0 or now - self.first_seen < self.max_feature_updates)
        if updating:
            self.median + x
            self.current_median = self.median.median()
        else:
            self.updating = False
        median = self.current_median

        if self.min is None:
            if x < median and (median - x) / median > self.threshold:
                if updating:
                    self.min = x
                    self.hard_min = self.min + self.hard_factor * (median - self.min)
                return -1
            return 0
        else:
            if x < self.min:
                if updating:
                    self.min = x
                    self.hard_min = self.min + self.hard_factor * (median - self.min)
                return -1
        if self.max is None:
            if x > median and (x - median) / median > self.threshold:
                if updating:
                    self.max = x
                    self.hard_max = self.max - self.hard_factor * (self.max - median)
                return 1
            return 0
        else:
            if x > self.max:
                if updating:
                    self.max = x
                    self.hard_max = self.max - self.hard_factor * (self.max - median)
                return 1

        if updating:
            if self.min < self.hard_min:
                self.min = self.hard_min * self.decay + self.min * (1 - self.decay)
            if self.max > self.hard_max:
                self.max = self.hard_max * self.decay + self.max * (1 - self.decay)

        if x < median:
            return - (1 - (x - self.min) / (median - self.min))
        elif x > median:
            return (x - median) / (self.max - median)

        return 0

class FeatureExtractor():
    def __init__(self, max_feature_updates=0):
        self.eye_l = Feature(max_feature_updates=max_feature_updates)
        self.eye_r = Feature(max_feature_updates=max_feature_updates)
        self.eyebrow_updown_l = Feature(max_feature_updates=max_feature_updates)
        self.eyebrow_updown_r = Feature(max_feature_updates=max_feature_updates)
        self.eyebrow_quirk_l = Feature(threshold=0.05, max_feature_updates=max_feature_updates)
        self.eyebrow_quirk_r = Feature(threshold=0.05, max_feature_updates=max_feature_updates)
        self.eyebrow_steepness_l = Feature(threshold=0.05, max_feature_updates=max_feature_updates)
        self.eyebrow_steepness_r = Feature(threshold=0.05, max_feature_updates=max_feature_updates)
        self.mouth_corner_updown_l = Feature(max_feature_updates=max_feature_updates)
        self.mouth_corner_updown_r = Feature(max_feature_updates=max_feature_updates)
        self.mouth_corner_inout_l = Feature(threshold=0.02, max_feature_updates=max_feature_updates)
        self.mouth_corner_inout_r = Feature(threshold=0.02, max_feature_updates=max_feature_updates)
        self.mouth_open = Feature(max_feature_updates=max_feature_updates)
        self.mouth_wide = Feature(threshold=0.02, max_feature_updates=max_feature_updates)

    def align_points(self, a, b, pts):
        a = tuple(a)
        b = tuple(b)
        alpha = angle(a, b)
        alpha = np.rad2deg(alpha)
        if alpha >= 90:
            alpha = - (alpha - 180)
        if alpha <= -90:
            alpha = - (alpha + 180)
        alpha = np.deg2rad(alpha)
        aligned_pts = []
        for pt in pts:
            aligned_pts.append(np.array(rotate(a, pt, alpha)))
        return alpha, np.array(aligned_pts)

    def update(self, pts, full=True):
        features = {}
        now = time.perf_counter()

        norm_distance_x = np.mean([pts[127, 0] - pts[356, 0], pts[234, 0] - pts[454, 0]])
        norm_distance_y = np.mean([pts[6, 1] - pts[197, 1], pts[197, 1] - pts[195, 1], pts[195, 1] - pts[5, 1]])

        a1, f_pts = self.align_points(pts[33], pts[133], pts[[160,158,144,153]])
        f = abs((np.mean([f_pts[0,1], f_pts[1,1]]) - np.mean([f_pts[2,1], f_pts[3,1]])) / norm_distance_y)
        features["eye_l"] = self.eye_l.update(f, now)

        a2, f_pts = self.align_points(pts[362], pts[263], pts[[385, 387, 380, 373]])
        f = abs((np.mean([f_pts[0,1], f_pts[1,1]]) - np.mean([f_pts[2,1], f_pts[3,1]])) / norm_distance_y)
        features["eye_r"] = self.eye_r.update(f, now)

        if full:
            a3, _ = self.align_points(pts[127], pts[356], [])
            a4, _ = self.align_points(pts[131], pts[360], [])
            norm_angle = np.mean(list(map(np.rad2deg, [a1, a2, a3, a4])))

            a, f_pts = self.align_points(pts[22], pts[26], pts[[22, 23, 24, 25, 26]])
            features["eyebrow_steepness_l"] = self.eyebrow_steepness_l.update(-np.rad2deg(a) - norm_angle, now)
            f = np.max(np.abs(np.array(f_pts[1:4]) - f_pts[0, 1])) / norm_distance_y
            features["eyebrow_quirk_l"] = self.eyebrow_quirk_l.update(f, now)

            a, f_pts = self.align_points(pts[17], pts[21], pts[[17, 18, 19, 20, 21]])
            features["eyebrow_steepness_r"] = self.eyebrow_steepness_r.update(np.rad2deg(a) - norm_angle, now)
            f = np.max(np.abs(np.array(f_pts[1:4]) - f_pts[0, 1])) / norm_distance_y
            features["eyebrow_quirk_r"] = self.eyebrow_quirk_r.update(f, now)
        else:
            features["eyebrow_steepness_l"] = 0.
            features["eyebrow_steepness_r"] = 0.
            features["eyebrow_quirk_l"] = 0.
            features["eyebrow_quirk_r"] = 0.

        f = (np.mean([pts[285, 1], pts[276, 1]]) - pts[168, 1]) / norm_distance_y
        features["eyebrow_updown_l"] = self.eyebrow_updown_l.update(f, now)

        f = (np.mean([pts[46, 1], pts[55, 1]]) - pts[168, 1]) / norm_distance_y
        features["eyebrow_updown_r"] = self.eyebrow_updown_r.update(f, now)

        upper_mouth_line = np.mean([pts[37, 1], pts[0, 1], pts[267, 1]])
        center_line = np.mean([pts[6, 0], pts[5, 0], pts[4, 0], pts[0, 0], pts[13, 0], pts[14, 0], pts[17, 0]])

        f = (upper_mouth_line - pts[62, 1]) / norm_distance_y
        features["mouth_corner_updown_l"] = self.mouth_corner_updown_l.update(f, now)
        if full:
            f = abs(center_line - pts[62, 0]) / norm_distance_x
            features["mouth_corner_inout_l"] = self.mouth_corner_inout_l.update(f, now)
        else:
            features["mouth_corner_inout_l"] = 0.

        f = (upper_mouth_line - pts[58, 1]) / norm_distance_y
        features["mouth_corner_updown_r"] = self.mouth_corner_updown_r.update(f, now)
        if full:
            f = abs(center_line - pts[58, 0]) / norm_distance_x
            features["mouth_corner_inout_r"] = self.mouth_corner_inout_r.update(f, now)
        else:
            features["mouth_corner_inout_r"] = 0.

        f = abs(np.mean(pts[[59,60,61], 1], axis=0) - np.mean(pts[[63,64,65], 1], axis=0)) / norm_distance_y
        features["mouth_open"] = self.mouth_open.update(f, now)

        f = abs(pts[58, 0] - pts[62, 0]) / norm_distance_x
        features["mouth_wide"] = self.mouth_wide.update(f, now)

        return features

class FaceInfo():
    def __init__(self, id, tracker):
        self.id = id
        self.frame_count = -1
        self.tracker = tracker
        self.reset()
        self.alive = False
        self.coord = None
        self.base_scale_v = 1
        self.base_scale_h = 1

        self.limit_3d_adjustment = True
        self.update_count_delta = 75.
        self.update_count_max = 7500.

        if self.tracker.max_feature_updates > 0:
            self.features = FeatureExtractor(self.tracker.max_feature_updates)

    def reset(self):
        self.alive = False
        self.conf = None
        self.lms = None
        self.eye_state = None
        self.rotation = None
        self.translation = None
        self.success = None
        self.quaternion = None
        self.euler = None
        self.pnp_error = None
        self.pts_3d = None
        self.eye_blink = None
        self.bbox = None
        self.holistic_info = None
        self.pnp_error = 0
        if self.tracker.max_feature_updates < 1:
            self.features = FeatureExtractor(0)
        self.current_features = {}
        self.contour = np.zeros((21,3))
        self.update_counts = np.zeros((66,2))
        self.fail_count = 0

    def update(self, result, frame_count):
        self.frame_count = frame_count
        if result is None:
            self.reset()
        else:
            self.holistic_info = result
            self.alive = True
            
    def adjust_3d(self):
        self.pts_3d = self.normalize_pts3d(self.holistic_info.face_landmarks)
        self.current_features = self.features.update(self.pts_3d[:, 0:2])
        self.eye_blink = []
        self.eye_blink.append(1 - min(max(0, -self.current_features["eye_r"]), 1))
        self.eye_blink.append(1 - min(max(0, -self.current_features["eye_l"]), 1))

    def normalize_pts3d(self, landmarks):
        print(len(landmarks.landmark))
        pts_3d = np.array([[l.x, l.y, l.z] for l in landmarks.landmark])
        # Calculate angle using nose
        pts_3d[:, 0:2] -= pts_3d[30, 0:2]
        alpha = angle(pts_3d[30, 0:2], pts_3d[27, 0:2])
        alpha -= np.deg2rad(90)

        R = np.matrix([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
        pts_3d[:, 0:2] = (pts_3d - pts_3d[30])[:, 0:2].dot(R) + pts_3d[30, 0:2]

        # Vertical scale
        pts_3d[:, 1] /= np.mean((pts_3d[27:30, 1] - pts_3d[28:31, 1]) / self.base_scale_v)

        # Horizontal scale
        pts_3d[:, 0] /= np.mean(np.abs(pts_3d[[0, 36, 42], 0] - pts_3d[[16, 39, 45], 0]) / self.base_scale_h)

        return pts_3d

def get_model_base_path(model_dir):
    model_base_path = resolve(os.path.join("models"))
    if model_dir is None:
        if not os.path.exists(model_base_path):
            model_base_path = resolve(os.path.join("..", "models"))
    else:
        model_base_path = model_dir
    return model_base_path

class Tracker():
    def __init__(self, width, height, model_type=3, detection_threshold=0.6, threshold=None, max_faces=1, discard_after=5, scan_every=3, bbox_growth=0.0, max_threads=4, silent=False, model_dir=None, no_gaze=False, use_retinaface=False, max_feature_updates=0, static_model=False, feature_level=2, try_hard=False):
        self.model_type = model_type
        
        self.holistic = mp_holistic.Holistic(model_complexity=1,min_detection_confidence=0.82,min_tracking_confidence=0.82,enable_segmentation=False,refine_face_landmarks=True)
        # self.face = mp_face_mesh.Face_Mesh
        
        model_base_path = get_model_base_path(None)
        model = os.path.join(model_base_path, "face_landmarker.task")
        
        with open(model, 'rb') as f:
            vision_model = f.read()
        
        base_options = python.BaseOptions(model_asset_buffer=vision_model)
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                                output_face_blendshapes=True,
                                                output_facial_transformation_matrixes=True,
                                                num_faces=1)
        self.detector = vision.FaceLandmarker.create_from_options(options)

        if threshold is None:
            threshold = 0.6
            if model_type < 0:
                threshold = 0.87

        self.faces = []

        # Image normalization constants
        self.mean = np.float32(np.array([0.485, 0.456, 0.406]))
        self.std = np.float32(np.array([0.229, 0.224, 0.225]))
        self.mean = self.mean / self.std
        self.std = self.std * 255.0

        self.mean = - self.mean
        self.std = 1.0 / self.std
        self.mean_32 = np.tile(self.mean, [32, 32, 1])
        self.std_32 = np.tile(self.std, [32, 32, 1])
        self.mean_224 = np.tile(self.mean, [224, 224, 1])
        self.std_224 = np.tile(self.std, [224, 224, 1])

        self.camera = np.array([[width, 0, width/2], [0, width, height/2], [0, 0, 1]], np.float32)
        self.inverse_camera = np.linalg.inv(self.camera)
        self.dist_coeffs = np.zeros((4,1))

        self.frame_count = 0
        self.width = width
        self.height = height
        self.threshold = threshold
        self.detection_threshold = detection_threshold
        self.max_faces = max_faces
        self.max_threads = max_threads
        self.discard = 0
        self.discard_after = discard_after
        self.detected = 0
        self.wait_count = 0
        self.scan_every = scan_every
        self.bbox_growth = bbox_growth
        self.silent = silent
        self.try_hard = try_hard

        self.res = 224.
        self.mean_res = self.mean_224
        self.std_res = self.std_224
        if model_type < 0:
            self.res = 56.
            self.mean_res = np.tile(self.mean, [56, 56, 1])
            self.std_res = np.tile(self.std, [56, 56, 1])
        if model_type < -1:
            self.res = 112.
            self.mean_res = np.tile(self.mean, [112, 112, 1])
            self.std_res = np.tile(self.std, [112, 112, 1])
        self.res_i = int(self.res)
        self.out_res = 27.
        if model_type < 0:
            self.out_res = 6.
        if model_type < -1:
            self.out_res = 13.
        self.out_res_i = int(self.out_res) + 1
        self.logit_factor = 16.
        if model_type < 0:
            self.logit_factor = 8.
        if model_type < -1:
            self.logit_factor = 16.

        self.no_gaze = no_gaze
        self.debug_gaze = False
        self.feature_level = feature_level
        if model_type == -1:
            self.feature_level = min(feature_level, 1)
        self.max_feature_updates = max_feature_updates
        self.static_model = static_model
        self.face_info = FaceInfo(id, self)
        self.fail_count = 0

    def preprocess(self, im, crop):
        x1, y1, x2, y2 = crop
        im = np.float32(im[y1:y2, x1:x2,::-1]) # Crop and BGR to RGB
        im = cv2.resize(im, (self.res_i, self.res_i), interpolation=cv2.INTER_LINEAR) * self.std_res + self.mean_res
        im = np.expand_dims(im, 0)
        im = np.transpose(im, (0,3,1,2))
        return im

    def equalize(self, im):
        im_yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
        im_yuv[:,:,0] = cv2.equalizeHist(im_yuv[:,:,0])
        return cv2.cvtColor(im_yuv, cv2.COLOR_YUV2BGR)

    def predict(self, frame):
        self.frame_count += 1
        start = time.perf_counter()
        im = frame
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=im)
        detection_result = self.detector.detect(mp_image)
        
        # results = self.holistic.process(im)
        
        duration = (time.perf_counter() - start) * 1000
        
        print(detection_result.face_blendshapes[0][25]);
        if not self.silent:
            print(f"Took {duration:.2f}ms")
        # results = sorted(results, key=lambda x: x.id)
        
        # self.face_info.update(results, self.frame_count)
        # self.face_info.adjust_3d()

        return self.face_info
