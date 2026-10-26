import sys, torch
sys.path.insert(0, sys.argv[1]); from srvgg import load
m=load(0.5, sys.argv[2]+'/models')
x=torch.rand(1,3,64,64)
torch.onnx.export(m, x, sys.argv[2]+'/models/srvgg_dn05.onnx', input_names=['x'], output_names=['y'], dynamic_axes={'x':{2:'h',3:'w'},'y':{2:'H',3:'W'}}, opset_version=17, dynamo=False)
print('ok')
