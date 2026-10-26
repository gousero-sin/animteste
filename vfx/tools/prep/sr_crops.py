import sys, os, numpy as np, cv2, onnxruntime as ort, time
S=sys.argv[1]; a,b=int(sys.argv[2]),int(sys.argv[3]); th=int(sys.argv[4])
so=ort.SessionOptions(); so.intra_op_num_threads=th
sess=ort.InferenceSession(f'{S}/models/srvgg_dn05.onnx',so,providers=['CPUExecutionProvider'])
N=2265; H0,W0=480,360; SC=3
def bbox(i):
    m=cv2.imread(f'{S}/alpha_lo/{i:04d}.png',0)>25
    ys,xs=np.where(m)
    if len(xs)==0: return None
    return xs.min(),ys.min(),xs.max(),ys.max()
boxes=[bbox(i) for i in range(N)]
t0=time.time()
for i in range(a,b):
    out=f'{S}/hyb/{i:04d}.jpg'
    if os.path.exists(out): continue
    bs=[boxes[j] for j in range(max(0,i-2),min(N,i+3)) if boxes[j] is not None]
    x0=min(bb[0] for bb in bs)-20; y0=min(bb[1] for bb in bs)-20; x1=max(bb[2] for bb in bs)+21; y1=max(bb[3] for bb in bs)+21
    x0=max(0,x0); y0=max(0,y0); x1=min(W0,x1); y1=min(H0,y1)
    x0-=x0%2; y0-=y0%2
    src=cv2.imread(f'{S}/allframes/{i:04d}.png')
    full=cv2.resize(src,(W0*SC,H0*SC),interpolation=cv2.INTER_LANCZOS4).astype(np.float32)/255
    crop=src[y0:y1,x0:x1,::-1].astype(np.float32)/255
    y=sess.run(None,{'x':crop.transpose(2,0,1)[None].copy()})[0][0].transpose(1,2,0)[:,:,::-1]
    y=np.clip(y,0,1)
    h,w=(y1-y0)*SC,(x1-x0)*SC
    y=cv2.resize(y,(w,h),interpolation=cv2.INTER_AREA)
    # feather mask: 1 inside, ramps to 0 over 30px at borders not on frame edge
    fm=np.ones((h,w),np.float32); F=30
    ramp=np.linspace(0,1,F,dtype=np.float32)
    if x0>0: fm[:,:F]*=ramp[None,:]
    if x1<W0: fm[:,-F:]*=ramp[::-1][None,:]
    if y0>0: fm[:F,:]*=ramp[:,None]
    if y1<H0: fm[-F:,:]*=ramp[::-1][:,None]
    reg=full[y0*SC:y1*SC,x0*SC:x1*SC]
    full[y0*SC:y1*SC,x0*SC:x1*SC]=reg*(1-fm[...,None])+y*fm[...,None]
    cv2.imwrite(out+'.tmp.jpg',(full*255).round().astype(np.uint8),[cv2.IMWRITE_JPEG_QUALITY,96,cv2.IMWRITE_JPEG_SAMPLING_FACTOR,cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444])
    os.rename(out+'.tmp.jpg',out)
    if i%100==0: print(i,time.time()-t0,flush=True)
print('done',flush=True)
