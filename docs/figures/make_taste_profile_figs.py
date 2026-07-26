#!/usr/bin/env python3
"""Figures for docs/taste-profile.* — model-interpretation illustrations.
Run: predictors/.venv/bin/python docs/figures/make_taste_profile_figs.py <sae_job_id> [en|es]
Outputs docs/figures/taste-*[-es].pdf (hyphenated; leaves make_taste_figures.py's taste_*.pdf alone)."""
import sys, json, importlib
import numpy as np, torch
from collections import Counter, defaultdict
from pathlib import Path
from sklearn.cluster import KMeans
from scipy.stats import gaussian_kde
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
DS=ROOT/"data/datasets/seq-20260718-222807"; MD=ROOT/"data/models/best-seq-nexttrack-20260718-230948-d3d4a"
SAE=ROOT/"data/interp"/sys.argv[1]/"model-sae.json"
LANG=sys.argv[2] if len(sys.argv)>2 else "en"; SUF="" if LANG=="en" else "-es"
sys.path.insert(0,str(ROOT/"predictors")); S=importlib.import_module("seq_nexttrack")
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":120})
CL={"electronic":"#2ca8a0","indie":"#7b5cd6","argentine":"#e08a1e","metal":"#d1495b","other":"#9aa0a8"}
TX={
 "land_title":{"en":"Your taste as a landscape — each dot is one listening session;\nvalleys (dark) are the modes you settle into",
               "es":"Tu gusto como un paisaje — cada punto es una sesión de escucha;\nlos valles (oscuros) son los modos en los que te instalás"},
 "land_x":{"en":"a 2-D shadow of the model's 192-D 'sound space'","es":"una 'sombra' 2-D del espacio de sonido de 192 dimensiones del modelo"},
 "imp_x":{"en":"internal directions the model devotes (atom count) = structural importance","es":"direcciones internas que le dedica el modelo (nº de átomos) = importancia estructural"},
 "imp_title":{"en":"How central each artist is *to the model*","es":"Qué tan central es cada artista *para el modelo*"},
 "bb_x":{"en":"Markov 'stay' rate  (co-occurrence model)","es":"tasa de permanencia de Markov  (modelo de co-ocurrencia)"},
 "bb_y":{"en":"GRU state-jump ‖Δh‖ on arrival  (neural gates)","es":"salto de estado ‖Δh‖ de la GRU al llegar  (compuertas neuronales)"},
 "bb_title":{"en":"Two independent models agree: blocks vs. bridges","es":"Dos modelos independientes coinciden: bloques vs. puentes"},
 "bb_blocks":{"en":"BLOCKS","es":"BLOQUES"},"bb_bridges":{"en":"BRIDGES","es":"PUENTES"},
 "flow_title":{"en":"The flow-grammar: how genres hand off (node size = 'hub' score)","es":"La gramática de flujo: cómo se pasan la posta los géneros (tamaño = 'hub')"},
 "att_title":{"en":"Gravity wells: roll the model forward from one seed →\nwhere it settles (green = stays home, red = flows elsewhere)","es":"Pozos gravitacionales: hacé rodar el modelo desde una semilla →\ndónde se asienta (verde = se queda, rojo = fluye a otro lado)"},
 "att_x":{"en":"share of a 30-step roll-out that lands in the destination genre","es":"proporción de un recorrido de 30 pasos que cae en el género destino"},
 "leg":{"en":["electronic","indie","argentine","metal"],"es":["electrónica","indie","argentino","metal"]},
}
def t(k): return TX[k][LANG]
def gcl(g):
    g=g or ""
    if any(k in g for k in ["argentin","electronica argentina","folklore","nuevo folk"]): return "argentine"
    if "metal" in g or "german" in g: return "metal"
    if any(k in g for k in ["house","techno","beat","dance","electro","idm","downtempo","garage","ambient","dusseldorf","trip hop","organic","balearic","electronica"]): return "electronic"
    if any(k in g for k in ["indie","rock","art pop","shoegaz","dream","post-rock","brooklyn"]): return "indie"
    return "other"
ARTCL={a:"electronic" for a in ["Four Tet","Caribou","Daphni","Aphex Twin","Underworld","Joy Orbison","Jamie xx","DJ Koze","The Avalanches","Massive Attack","Thievery Corporation","Air","Tosca","Kraftwerk","Moby","Daft Punk","Gorillaz"]}
ARTCL.update({a:"indie" for a in ["DIIV","The xx","Radiohead"]}); ARTCL.update({a:"argentine" for a in ["Laika Perra Rusa","Peces Raros","Babasonicos","Usted Señalemelo"]}); ARTCL["Rammstein"]="metal"
def legend(ax,loc):
    labs=t("leg")
    ax.legend(handles=[plt.Line2D([0],[0],marker="s",ls="",color=CL[k],label=l) for k,l in zip(["electronic","indie","argentine","metal"],labs)],fontsize=8,loc=loc,frameon=False)

items=json.load(open(DS/"items.json")); mx=max(int(k) for k in items)+1
gen=[None]*mx; art=[None]*mx
for k,v in items.items():
    i=int(k)
    if i<mx: gen[i]=v.get("genre"); art[i]=v.get("artist")
lat=np.fromfile(DS/"item_latents.f32",dtype=np.float32).reshape(-1,192)
sess=np.fromfile(DS/"sessions.u32",dtype=np.uint32); off=np.fromfile(DS/"offsets.u32",dtype=np.uint32)
sessions=[sess[off[s]:off[s+1]].astype(int) for s in range(len(off)-1)]

# ---- landscape ----
cent=np.array([ (lambda v: v/(np.linalg.norm(v)+1e-9))(lat[idx].mean(0)) for idx in sessions],np.float32)
K=6; lab=KMeans(K,n_init=8,random_state=0).fit_predict(cent)
M=cent-cent.mean(0); _,_,Vt=np.linalg.svd(M,full_matrices=False); P=M@Vt[:2].T
mode_lab={c:Counter(art[i] for s in np.where(lab==c)[0] for i in sessions[s] if art[i]).most_common(1)[0][0] for c in range(K)}
fig,ax=plt.subplots(figsize=(7.4,6.0))
kde=gaussian_kde(np.vstack([P[:,0],P[:,1]]))
gx,gy=np.mgrid[P[:,0].min():P[:,0].max():160j,P[:,1].min():P[:,1].max():160j]
ax.contourf(gx,gy,kde(np.vstack([gx.ravel(),gy.ravel()])).reshape(gx.shape),levels=12,cmap="Greys",alpha=0.55)
modecol={}
for c in range(K):
    m=lab==c; col=CL[gcl(Counter(g for s in np.where(m)[0] for g in [gen[i] for i in sessions[s]] if g).most_common(1)[0][0])]
    modecol[c]=col; ax.scatter(P[m,0],P[m,1],s=5,color=col,alpha=0.35,linewidths=0)
for c in range(K):
    ax.annotate(f"{mode_lab[c]}\n({(lab==c).mean()*100:.0f}%)",(np.median(P[lab==c,0]),np.median(P[lab==c,1])),fontsize=9,fontweight="bold",ha="center",va="center",bbox=dict(boxstyle="round,pad=0.25",fc="white",ec=modecol[c],alpha=0.9))
ax.set_title(t("land_title"),loc="left",fontweight="bold",fontsize=11); ax.set_xlabel(t("land_x")); ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout(); fig.savefig(OUT/f"taste-landscape{SUF}.pdf"); plt.close(fig)

# ---- importance ----
abc=json.load(open(SAE))["layers"][0]["atoms_by_concept"]; cnt=Counter(a["concept_value"] for a in abc if a["concept_field"]=="artist"); top=cnt.most_common(20)[::-1]
fig,ax=plt.subplots(figsize=(6.4,5.4))
ax.barh([a for a,_ in top],[c for _,c in top],color=[CL[ARTCL.get(a,"other")] for a,_ in top])
for i,(a,c) in enumerate(top): ax.text(c+0.3,i,str(c),va="center",fontsize=8)
ax.set_xlabel(t("imp_x")); ax.set_title(t("imp_title"),loc="left",fontweight="bold"); legend(ax,"lower right")
fig.tight_layout(); fig.savefig(OUT/f"taste-importance{SUF}.pdf"); plt.close(fig)

# ---- transitions + Δh ----
gt=Counter(); gself=Counter(); gtot=Counter(); starts=Counter(); ends=Counter()
for idx in sessions:
    gg=[gen[i] for i in idx if gen[i]]
    if gg: starts[gg[0]]+=1; ends[gg[-1]]+=1
    for a,b in zip(gg,gg[1:]):
        gtot[a]+=1
        if a==b: gself[a]+=1
        else: gt[(a,b)]+=1
stay={g:gself[g]/gtot[g] for g in gtot if gtot[g]>=300}
hp=S.load_hp(str(MD/"hyperparams.json")); artf=S.load_artifact(DS); model=S.load_model(MD,artf,hp); model.eval()
gru=model.rnn;H=gru.hidden_size
Wih=gru.weight_ih_l0.detach().numpy();Whh=gru.weight_hh_l0.detach().numpy();bih=gru.bias_ih_l0.detach().numpy();bhh=gru.bias_hh_l0.detach().numpy()
sg=lambda x:1/(1+np.exp(-x)); dhg=defaultdict(list); test=np.fromfile(DS/"test_sessions.u32",dtype=np.uint32)
for s in test:
    idx=sessions[s]
    if len(idx)<3: continue
    X=lat[idx];h=np.zeros(H);prev=None
    for tt in range(len(idx)):
        gi=Wih@X[tt]+bih;gh=Whh@h+bhh;r=sg(gi[:H]+gh[:H]);z=sg(gi[H:2*H]+gh[H:2*H]);n=np.tanh(gi[2*H:]+r*gh[2*H:]);hn=(1-z)*n+z*h
        if prev is not None and gen[idx[tt]]: dhg[gen[idx[tt]]].append(np.linalg.norm(hn-h))
        h=hn;prev=1
dh={g:np.mean(v) for g,v in dhg.items() if len(v)>=200}; common=[g for g in stay if g in dh]
fig,ax=plt.subplots(figsize=(6.8,5.4))
for g in common: ax.scatter(stay[g],dh[g],s=36,color=CL[gcl(g)],zorder=3)
show={"melodic techno","deep house","jazz house","alternative dance","argentine rock","alternative metal","downtempo","big beat","art pop","uk garage","idm","trip hop","dusseldorf electronic","ambient","indie rock"}
for g in common:
    if g in show: ax.annotate(g,(stay[g],dh[g]),fontsize=7,xytext=(3,3),textcoords="offset points")
ax.set_xlabel(t("bb_x")); ax.set_ylabel(t("bb_y")); ax.set_title(t("bb_title"),loc="left",fontweight="bold")
ax.annotate(t("bb_blocks"),(max(stay[g] for g in common)*0.86,min(dh[g] for g in common)*1.03),color="#444",fontweight="bold",ha="center")
ax.annotate(t("bb_bridges"),(min(stay[g] for g in common)*1.15,max(dh[g] for g in common)*0.98),color="#444",fontweight="bold",ha="center")
fig.tight_layout(); fig.savefig(OUT/f"taste-blocks-bridges{SUF}.pdf"); plt.close(fig)

main=["argentine indie","argentine alternative rock","argentine rock","electronica argentina","alternative dance","downtempo","big beat","electronica","ambient","alternative rock","alternative metal","dusseldorf electronic","uk garage","album rock"]
G=nx.DiGraph()
for (a,b),c in gt.items():
    if a in main and b in main and c>=55: G.add_edge(a,b,weight=c)
pr=nx.pagerank(G,weight="weight"); pos=nx.spring_layout(G,seed=4,k=1.2,iterations=250)
fig,ax=plt.subplots(figsize=(7.4,5.8))
nx.draw_networkx_nodes(G,pos,node_color=[CL[gcl(g)] for g in G.nodes],node_size=[300+9000*pr[g] for g in G.nodes],alpha=0.9,ax=ax)
ws=np.array([G[u][v]["weight"] for u,v in G.edges]); nx.draw_networkx_edges(G,pos,width=0.4+2.6*ws/ws.max(),edge_color="#b9bdc4",arrows=True,arrowsize=9,connectionstyle="arc3,rad=0.09",ax=ax)
nx.draw_networkx_labels(G,pos,font_size=7.5,ax=ax)
ax.set_title(t("flow_title"),loc="left",fontweight="bold"); ax.axis("off")
fig.tight_layout(); fig.savefig(OUT/f"taste-flow{SUF}.pdf"); plt.close(fig)

# ---- attractors ----
Ln=lat/(np.linalg.norm(lat,axis=1,keepdims=True)+1e-9)
def pred(seq):
    with torch.no_grad(): enc=model._encode_seq(torch.from_numpy(lat[seq][None].astype(np.float32)),None)[0]; nl=model.head(enc[-1]).numpy()
    return nl/(np.linalg.norm(nl)+1e-9)
first={}
for i in range(mx):
    if art[i] and art[i] not in first: first[art[i]]=i
seeds=["Kraftwerk","Aphex Twin","Rammstein","Radiohead","Babasonicos","Peces Raros","Joy Orbison","Massive Attack","Four Tet","The xx","DIIV","Underworld","Caribou","Daft Punk"]
res=[]
for a in seeds:
    if a not in first: continue
    seq=[first[a]];used={first[a]};arts={a};gc=Counter()
    for _ in range(30):
        p=pred(seq);sims=Ln@p
        for r in used: sims[r]=-9
        for r in np.argsort(-sims)[:60]:
            if art[r] in arts: sims[r]-=0.5
        r=int(np.argmax(sims));seq.append(r);used.add(r);arts.add(art[r])
        if gen[r]: gc[gen[r]]+=1
    dom,dn=gc.most_common(1)[0]; res.append((a,gen[first[a]],dom,dn/sum(gc.values())))
res=res[::-1]
fig,ax=plt.subplots(figsize=(7.2,5.6))
for i,(a,sg_,dom,sh) in enumerate(res):
    ax.barh(i,sh,color=CL["metal"] if dom!=sg_ else CL["electronic"],alpha=0.85)
    ax.text(0.01,i,a,va="center",ha="left",fontsize=8.5,fontweight="bold"); ax.text(sh+0.01,i,f"→ {dom}",va="center",fontsize=8)
ax.set_yticks([]); ax.set_xlim(0,1.75); ax.set_xlabel(t("att_x")); ax.set_title(t("att_title"),loc="left",fontweight="bold",fontsize=10.5)
fig.tight_layout(); fig.savefig(OUT/f"taste-attractors{SUF}.pdf"); plt.close(fig)
print("wrote", LANG, *[p.name for p in sorted(OUT.glob(f"taste-*{SUF}.pdf")) if (SUF or "-es" not in p.name)])
