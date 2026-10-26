import sys, numpy as np, cv2, onnxruntime as ort, time
S=sys.argv[1]; model=sys.argv[2]; dr=float(sys.argv[3]); outdir=sys.argv[4]
so=ort.SessionOptions(); so.intra_op_num_threads=int(sys.argv[5])
sess=ort.InferenceSession(f'{S}/models/{model}', so, providers=['CPUExecutionProvider'])
rec=[np.zeros([1,1,1,1],np.float32)]*4
d=np.array([dr],np.float32)
t0=time.time()
for i in range(2265):
    im=cv2.imread(f'{S}/allframes/{i:04d}.png')[:,:,::-1].astype(np.float32)/255
    x=im.transpose(2,0,1)[None].copy()
    fgr,pha,*rec=sess.run([], {'src':x,'r1i':rec[0],'r2i':rec[1],'r3i':rec[2],'r4i':rec[3],'downsample_ratio':d})
    cv2.imwrite(f'{outdir}/{i:04d}.png',(pha[0,0]*255).round().astype(np.uint8))
    if i%300==0: print(i,time.time()-t0,flush=True)
print('done',flush=True)
