import sys, os, numpy as np, cv2, onnxruntime as ort, time
S=sys.argv[1]; model=sys.argv[2]; dr=float(sys.argv[3]); a,b=int(sys.argv[4]),int(sys.argv[5]); th=int(sys.argv[6])
so=ort.SessionOptions(); so.intra_op_num_threads=th
sess=ort.InferenceSession(f'{S}/models/{model}', so, providers=['CPUExecutionProvider'])
rec=[np.zeros([1,1,1,1],np.float32)]*4
d=np.array([dr],np.float32)
t0=time.time()
# warm-up the recurrent state on a few frames before `a` if starting mid-video
start=max(0,a-30) if a>0 else 0
for i in range(start,b):
    p=f'{S}/hyb/{i:04d}.jpg'
    while not os.path.exists(p): time.sleep(5)
    im=cv2.imread(p)[:,:,::-1].astype(np.float32)/255
    x=im.transpose(2,0,1)[None].copy()
    fgr,pha,*rec=sess.run([], {'src':x,'r1i':rec[0],'r2i':rec[1],'r3i':rec[2],'r4i':rec[3],'downsample_ratio':d})
    if i<a: continue
    cv2.imwrite(f'{S}/alpha/{i:04d}.png',(pha[0,0]*255).round().astype(np.uint8))
    f=(np.clip(fgr[0].transpose(1,2,0),0,1)[:,:,::-1]*255).round().astype(np.uint8)
    cv2.imwrite(f'{S}/fgr/{i:04d}.jpg',f,[cv2.IMWRITE_JPEG_QUALITY,95,cv2.IMWRITE_JPEG_SAMPLING_FACTOR,cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444])
    if i%100==0: print(i,time.time()-t0,flush=True)
print('done',flush=True)
