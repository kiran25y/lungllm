import torch, numpy as np, pandas as pd
import lungllm.training.train_multitask as T
from lungllm.models.pretrained_encoder import ASTEncoder, PretrainedMultiTaskModel

ck = torch.load("checkpoints/sft/ast_multitask.pt", map_location="cpu", weights_only=False)
dv = ck.get("disease_vocab", {}); inv = {v:k for k,v in dv.items()}
model = PretrainedMultiTaskModel(ASTEncoder(freeze=False), num_anomaly=4, num_disease=len(dv))
model.load_state_dict(ck["model"]); model.to("cuda").eval()

# all mendeley rows with valid diagnosis
df = pd.read_parquet("data/processed/manifests/all.parquet")
df = df[df.dataset=="mendeley"].reset_index(drop=True)
tmp = "data/splits/_mendeley_all.parquet"; df.to_parquet(tmp, index=False)
ds = T.DS(tmp, dv)

# extract 768-d encoder embeddings
X, Y, G = [], [], []
with torch.no_grad():
    for i in range(len(ds)):
        w, _, didx = ds[i]
        if int(didx) == T.IGNORE: continue
        emb = model.encoder([w])["clip_embedding"].squeeze(0).cpu().numpy()
        X.append(emb); Y.append(int(didx)); G.append(df.iloc[i]["patient_id"])
X=np.array(X); Y=np.array(Y); G=np.array(G)

# keep classes with >=15 total support
keep = [c for c in set(Y.tolist()) if (Y==c).sum()>=15]
mask = np.isin(Y, keep); X,Y,G = X[mask],Y[mask],G[mask]
remap = {c:i for i,c in enumerate(sorted(keep))}; Yr = np.array([remap[c] for c in Y])
print("classes:", {inv.get(c,c):int((Y==c).sum()) for c in sorted(keep)})

# patient-level 5 folds
pats = np.array(sorted(set(G))); rng=np.random.RandomState(0); rng.shuffle(pats)
folds = np.array_split(pats, 5)
def probe(Xtr,ytr,Xte,nc,epochs=300):
    Xtr=torch.tensor(Xtr).float().cuda(); ytr=torch.tensor(ytr).long().cuda()
    clf=torch.nn.Linear(Xtr.shape[1], nc).cuda(); opt=torch.optim.Adam(clf.parameters(),1e-3,weight_decay=1e-4)
    lf=torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad(); loss=lf(clf(Xtr),ytr); loss.backward(); opt.step()
    with torch.no_grad(): return clf(torch.tensor(Xte).float().cuda()).argmax(-1).cpu().numpy()

nc=len(keep); mf1s=[]; baccs=[]
for k in range(5):
    teP=np.isin(G, folds[k]); trP=~teP
    if teP.sum()==0 or trP.sum()==0: continue
    pred=probe(X[trP],Yr[trP],X[teP],nc)
    gt=Yr[teP]
    f1s=[]; recs=[]
    for c in range(nc):
        tp=int(((pred==c)&(gt==c)).sum());fp=int(((pred==c)&(gt!=c)).sum());fn=int(((pred!=c)&(gt==c)).sum())
        pr=tp/(tp+fp) if tp+fp else 0; rc=tp/(tp+fn) if tp+fn else 0
        if (gt==c).sum()>0: f1s.append(2*pr*rc/(pr+rc) if pr+rc else 0); recs.append(rc)
    mf1=np.mean(f1s); bacc=np.mean(recs); mf1s.append(mf1); baccs.append(bacc)
    print(f"fold {k}: n_test={int(teP.sum())} macroF1={mf1:.3f} balAcc={bacc:.3f}")
print(f"\n== 5-fold patient-level (Mendeley, linear probe over trained encoder) ==")
print(f"macroF1 = {np.mean(mf1s):.3f} ± {np.std(mf1s):.3f}   balAcc = {np.mean(baccs):.3f} ± {np.std(baccs):.3f}")
