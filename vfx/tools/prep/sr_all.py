import sys, os, time, torch, numpy as np, cv2
sys.path.insert(0, sys.argv[1]); from srvgg import load
S=sys.argv[2]; a,b=int(sys.argv[3]),int(sys.argv[4]); torch.set_num_threads(int(sys.argv[5]))
m = load(0.5, S+'/models')
t0=time.time()
for i in range(a,b):
    out=f'{S}/sr/{i:04d}.jpg'
    if os.path.exists(out): continue
    im = cv2.imread(f'{S}/allframes/{i:04d}.png')[:,:,::-1].astype(np.float32)/255
    x = torch.from_numpy(im.copy()).permute(2,0,1)[None]
    with torch.no_grad(): y = m(x)[0].permute(1,2,0).clamp(0,1).numpy()
    y = cv2.resize(y, (1080,1440), interpolation=cv2.INTER_AREA)
    cv2.imwrite(out+'.tmp.jpg', (y[:,:,::-1]*255).round().astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 96, cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444])
    os.rename(out+'.tmp.jpg', out)
    if i%50==0: print(i, time.time()-t0, flush=True)
print('done', flush=True)
