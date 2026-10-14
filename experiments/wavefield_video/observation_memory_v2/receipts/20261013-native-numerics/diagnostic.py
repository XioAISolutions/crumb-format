"""Bounded negative evidence: sparse FFT causality/parity versus exact frame steps.
No training, GPU, downloads, native-source changes or tolerance relaxation.
"""
from pathlib import Path
import hashlib,json,sys,time
import torch

ROOT=Path(__file__).resolve().parents[2]
NATIVE=ROOT.parent
sys.path.insert(0,str(NATIVE))
from wfvideo import VideoPredictor

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def model(seed,layers=1):
    torch.manual_seed(seed)
    return VideoPredictor(16,layers,2,32,4,4,'wave',causal=True,residual=False,
      time_pos='none',kernel_version='dispersion',linear_pad=True,pole_param='halflife',
      hl_min=2.0,hl_max=4096.0,clean_write=True).eval()
def stream(m,x):
    st=m.stream_init(1,'cpu'); rows=[]
    for t in range(x.shape[1]):
        row,st=m.stream_step(x[:,t],st,t);rows.append(row)
    return torch.stack(rows,1),st

files=[NATIVE/'wfvideo.py',NATIVE/'ssm_lite.py',ROOT/'adapters.py',ROOT/'test_adapters.py']
before={str(p):sha(p) for p in files}
torch.set_num_threads(1)
t0=time.monotonic();cpu0=time.process_time()
x=torch.zeros(1,32,3,4,4);x[:,3,:,1,2]=torch.tensor([.18,-.14,.12])
y=x.clone();y[:,12:]=torch.randn(1,20,3,4,4,generator=torch.Generator().manual_seed(31))
poison=[];parity=[]
with torch.no_grad():
    for seed in (0,9,27,42,83):
        m=model(seed)
        a=m(x,states=[None],dense=True)[0];b=m(y,states=[None],dense=True)[0]
        diff=(a[:,:12]-b[:,:12]).abs()
        exact_a,_=stream(m,x);exact_b,_=stream(m,y)
        poison.append({'initialization_seed':seed,'dense_fft_prefix_max_delta':float(diff.max()),
          'dense_fft_blank_before_cue_max_delta':float(diff[:,:3].max()),
          'dense_fft_prefix_rms_delta':float(diff.square().mean().sqrt()),
          'dense_fft_allclose_atol3e_5_rtol3e_5':torch.allclose(a[:,:12],b[:,:12],atol=3e-5,rtol=3e-5),
          'exact_frame_prefix_equal':torch.equal(exact_a[:,:12],exact_b[:,:12])})
    z=torch.zeros_like(x);z[:,31,:,1,2]=torch.tensor([.18,-.14,.12])
    for layers in (1,2,4):
        m=model(42,layers)
        p,state=m(z,states=[None]*layers)
        q,frame_state=stream(m,z)
        parity.append({'initialization_seed':42,'layers':layers,
          'final_prediction_chunk_frame_max_delta':float((p-q[:,-1]).abs().max()),
          'recurrent_state_chunk_frame_max_delta':max(float((a-b).abs().max()) for a,b in zip(state,frame_state))})
    random=torch.randn(x.shape,generator=torch.Generator().manual_seed(991))
    random_poison=random.clone();random_poison[:,12:]=y[:,12:]
    m=model(42)
    a=m(random,states=[None],dense=True)[0];b=m(random_poison,states=[None],dense=True)[0]
    nonblank={'seed':991,'head_seed':42,'prefix_max_delta':float((a[:,:12]-b[:,:12]).abs().max())}
after={str(p):sha(p) for p in files}
if before!=after: raise RuntimeError('Read-only source changed during diagnostic; reject')
receipt={'protocol':'native_wave_sparse_numerics_negative_control_v1','device':'cpu','torch':torch.__version__,
 'threads':1,'source_sha256':before,'source_unchanged':True,
 'diagnostic_sha256':sha(Path(__file__)),'wall_seconds':time.monotonic()-t0,'cpu_seconds':time.process_time()-cpu0,
 'sparse_prefix_cue_frame':3,'poisoned_suffix_start':12,'poison_rng_seed':31,
 'shape':[1,32,3,4,4],'head':{'dim':16,'heads':2,'window':32,'pole_param':'halflife','clean_write':True,'residual':False},
 'prefix_poisoning':poison,'chunk_frame_parity_cue_frame':31,'chunk_frame_parity':parity,
 'nonblank_control':nonblank,
 'interpretation':'Temporal FFT mathematically truncated but sparse near-zero outputs can amplify numerical contamination through RMSNorm. The exact frame API is causally unchanged under suffix poisoning. Multilayer wave FFT/frame parity is unsupported; additive adapter restricts the wave candidate to one layer. No future target was ingested in final-only retrieval.',
 'trained':False,'gpu_used':False,'canonical_edited':False,'quality_promotion_allowed':False}
output=Path(__file__).with_name('receipt.json')
with output.open('x') as f:json.dump(receipt,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({'receipt':str(output),'source_unchanged':True,'sparse_max':[r['dense_fft_prefix_max_delta'] for r in poison],'parity_max':[r['final_prediction_chunk_frame_max_delta'] for r in parity],'exact_frame_prefix_equal_all':all(r['exact_frame_prefix_equal'] for r in poison)}))
