import sys, json, numpy as np, cv2, mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
S=sys.argv[1]; N=2265; FPS=1359000/45431
scale=2  # feed 2x lanczos frames for better small-hand detection
def opt(cls, path, **kw):
    return cls(base_options=BaseOptions(model_asset_path=path), running_mode=vision.RunningMode.VIDEO, **kw)
pose = vision.PoseLandmarker.create_from_options(opt(vision.PoseLandmarkerOptions, f'{S}/models/pose_landmarker_heavy.task', num_poses=1, min_pose_detection_confidence=0.3, min_tracking_confidence=0.3))
hands = vision.HandLandmarker.create_from_options(opt(vision.HandLandmarkerOptions, f'{S}/models/hand_landmarker.task', num_hands=2, min_hand_detection_confidence=0.2, min_hand_presence_confidence=0.2, min_tracking_confidence=0.2))
face = vision.FaceLandmarker.create_from_options(opt(vision.FaceLandmarkerOptions, f'{S}/models/face_landmarker.task', num_faces=1, min_face_detection_confidence=0.3, min_tracking_confidence=0.3))
out=[]
for i in range(N):
    im=cv2.imread(f'{S}/allframes/{i:04d}.png')
    im=cv2.resize(im,(360*scale,480*scale),interpolation=cv2.INTER_LANCZOS4)
    rgb=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    mi=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
    ts=int(round(i*1000/FPS))
    r={'i':i}
    p=pose.detect_for_video(mi,ts)
    if p.pose_landmarks:
        r['pose']=[[l.x,l.y,l.z,l.visibility] for l in p.pose_landmarks[0]]
    h=hands.detect_for_video(mi,ts)
    r['hands']=[{'lm':[[l.x,l.y,l.z] for l in lm],'hand':hd[0].category_name,'score':hd[0].score} for lm,hd in zip(h.hand_landmarks,h.handedness)]
    f=face.detect_for_video(mi,ts)
    if f.face_landmarks:
        r['face']=[[round(l.x,5),round(l.y,5),round(l.z,5)] for l in f.face_landmarks[0]]
    out.append(r)
    if i%200==0: print(i,flush=True)
json.dump(out,open(f'{S}/track/raw.json','w'))
print('done')
