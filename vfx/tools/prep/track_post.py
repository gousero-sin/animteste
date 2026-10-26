import json, numpy as np, sys
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
S=sys.argv[1]
d=json.load(open(f'{S}/track/raw.json'))
N=len(d); SC=3.0; W0,H0=360,480
OVAL=[10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
pose=np.full((N,33,4),np.nan)
for r in d:
    if 'pose' in r: pose[r['i']]=np.array(r['pose'])
pose[...,0]*=W0*SC; pose[...,1]*=H0*SC
# fill missing pose frames by interpolation (none expected)
for j in range(33):
    for c in range(4):
        v=pose[:,j,c]; m=np.isnan(v)
        if m.any() and (~m).any(): v[m]=np.interp(np.where(m)[0],np.where(~m)[0],v[~m])
pose_s=pose.copy()
for j in range(33):
    for c in range(2):
        pose_s[:,j,c]=savgol_filter(pose[:,j,c],9,2)
# hands: assign by proximity to pose wrists (16=R, 15=L)
hand={'R':np.full((N,21,3),np.nan),'L':np.full((N,21,3),np.nan)}
for r in d:
    i=r['i']; hs=r['hands']
    if not hs: continue
    pts=[np.array(h['lm'])*[W0*SC,H0*SC,W0*SC] for h in hs]
    wr={'R':pose[i,16,:2],'L':pose[i,15,:2]}
    if len(pts)==1:
        p=pts[0]; dR=np.linalg.norm(p[0,:2]-wr['R']); dL=np.linalg.norm(p[0,:2]-wr['L'])
        k='R' if dR<dL else 'L'
        if min(dR,dL)<250: hand[k][i]=p
    else:
        a,b=pts[:2]
        c1=np.linalg.norm(a[0,:2]-wr['R'])+np.linalg.norm(b[0,:2]-wr['L'])
        c2=np.linalg.norm(a[0,:2]-wr['L'])+np.linalg.norm(b[0,:2]-wr['R'])
        if c1<=c2: hand['R'][i]=a; hand['L'][i]=b
        else: hand['L'][i]=a; hand['R'][i]=b
det={k:~np.isnan(hand[k][:,0,0]) for k in hand}
# gap fill: for gaps <= 8 frames between detections interpolate; otherwise build from pose wrist offset of nearest detection (<= 20 frames), else leave nan
for k,wi in (('R',16),('L',15)):
    h=hand[k]; m=det[k]; idx=np.where(m)[0]
    if len(idx)==0: continue
    for i in range(N):
        if m[i]: continue
        prev=idx[idx<i]; nxt=idx[idx>i]
        p=prev[-1] if len(prev) else None; n=nxt[0] if len(nxt) else None
        if p is not None and n is not None and n-p<=8:
            a=(i-p)/(n-p); h[i]=h[p]*(1-a)+h[n]*a
        else:
            cand=[c for c in (p,n) if c is not None and abs(c-i)<=20]
            if cand:
                c=min(cand,key=lambda c:abs(c-i))
                off=pose_s[i,wi,:2]-pose_s[c,wi,:2]
                h[i]=h[c]; h[i,:,:2]=h[c,:,:2]+off
    # smooth along time within valid runs
    v=~np.isnan(h[:,0,0])
    runs=[]; s=None
    for i in range(N+1):
        if i<N and v[i]:
            if s is None: s=i
        elif s is not None: runs.append((s,i)); s=None
    for s,e in runs:
        if e-s>=5:
            h[s:e,:,:2]=gaussian_filter1d(h[s:e,:,:2],1.2,axis=0,mode='nearest')
face=np.full((N,478,3),np.nan)
for r in d:
    if 'face' in r: face[r['i']]=np.array(r['face'])*[W0*SC,H0*SC,W0*SC]
fdet=~np.isnan(face[:,0,0])
fi=np.where(fdet)[0]
for i in range(N):
    if not fdet[i]:
        # nearest detected face shifted by nose(0) movement in pose
        c=fi[np.argmin(np.abs(fi-i))]
        face[i]=face[c]; face[i,:,:2]+=pose_s[i,0,:2]-pose_s[c,0,:2]
face_s=face.copy()
face_s[:,:,:2]=gaussian_filter1d(face[:,:,:2],1.0,axis=0,mode='nearest')
np.savez_compressed(f'{S}/track/track.npz', pose=pose_s.astype(np.float32), pose_raw=pose.astype(np.float32),
    handR=hand['R'].astype(np.float32), handL=hand['L'].astype(np.float32), handR_det=det['R'], handL_det=det['L'],
    face=face_s[:,:,:2].astype(np.float32), face_det=fdet, face_oval_idx=np.array(OVAL))
print('R det',det['R'].sum(),'filled',(~np.isnan(hand['R'][:,0,0])).sum(),'L det',det['L'].sum(),'filled',(~np.isnan(hand['L'][:,0,0])).sum(),'face det',fdet.sum())
