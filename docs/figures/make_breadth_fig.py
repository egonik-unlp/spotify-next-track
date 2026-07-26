#!/usr/bin/env python3
"""Breadth / concentration of the artist distribution — the long tail the model
can't represent. Renders taste-breadth[-es].pdf. Run: ... make_breadth_fig.py [en|es]"""
import sys
import numpy as np
from collections import defaultdict
from pathlib import Path
from qdrant_client import QdrantClient
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

LANG=sys.argv[1] if len(sys.argv)>1 else "en"; SUF="" if LANG=="en" else "-es"
OUT=Path(__file__).resolve().parent
qc=QdrantClient(url="http://localhost:6337",timeout=180)
plays=defaultdict(float); off=None
while True:
    pts,off=qc.scroll("spotify_tracks_content",limit=4096,offset=off,with_payload=True,with_vectors=False)
    for p in pts:
        m=(p.payload or {}).get("metadata",{}) or {}
        a=m.get("artist"); pc=m.get("play_count")
        if a and isinstance(pc,(int,float)): plays[a]+=pc
    if off is None: break
v=np.array(sorted(plays.values()))            # ascending
N=len(v); tot=v.sum()
cum=np.cumsum(v)/tot
xa=np.arange(1,N+1)/N                          # cumulative share of artists
gini=1-2*np.trapezoid(cum,xa)
# head = top 25 artists (the model's "cast")
vd=np.sort(v)[::-1]; head=vd[:25].sum()/tot
one=int((v<=1).sum()); two=int((v<=2).sum())
print(f"artists={N} plays={int(tot)} gini={gini:.2f} top25_share={head:.2%} "
      f"artists<=1play={one}({one/N:.0%}) <=2={two}({two/N:.0%})")
# median artist
print(f"median artist plays={np.median(v):.0f} mean={v.mean():.1f}")

TX={"title":{"en":"You range widely, but thinly: the long tail the model can't see",
             "es":"Escuchás a lo ancho, pero fino: la larga cola que el modelo no ve"},
    "x":{"en":"share of your artists (poorest-listened → most)","es":"proporción de tus artistas (menos escuchados → más)"},
    "y":{"en":"share of all your plays","es":"proporción de todas tus reproducciones"},
    "eq":{"en":"perfect evenness","es":"reparto perfectamente parejo"},
    "note":{"en":f"{N:,} artists — half played once or twice.\nYour top 25 (0.4% of artists) = {head:.0%} of plays;\nthe SAE can only build atoms for that head.",
            "es":f"{N:,} artistas — la mitad escuchados una o dos veces.\nTus 25 top (0,4% de los artistas) = {head:.0%} de las reproducciones;\nel SAE solo puede armar átomos para esa cabeza."}}
fig,ax=plt.subplots(figsize=(6.6,5.0))
ax.plot([0,1],[0,1],"--",color="#b9bdc4",lw=1,label=TX["eq"][LANG])
ax.plot(np.concatenate([[0],xa]),np.concatenate([[0],cum]),color="#2ca8a0",lw=2.2)
ax.fill_between(np.concatenate([[0],xa]),np.concatenate([[0],cum]),np.concatenate([[0],xa]),color="#2ca8a0",alpha=0.12)
# mark the top-25 point (rightmost 25 artists)
x25=1-25/N; y25=cum[np.searchsorted(xa,x25)]
ax.scatter([x25],[y25],color="#d1495b",zorder=5)
ax.annotate(("top 25 →" if LANG=="en" else "top 25 →"),(x25,y25),xytext=(-4,10),textcoords="offset points",ha="right",fontsize=8,color="#d1495b",fontweight="bold")
ax.text(0.03,0.80,TX["note"][LANG],fontsize=8.5,va="top")
ax.text(0.55,0.06,f"Gini = {gini:.2f}",fontsize=10,fontweight="bold",color="#444")
ax.set_xlabel(TX["x"][LANG]); ax.set_ylabel(TX["y"][LANG]); ax.set_title(TX["title"][LANG],loc="left",fontweight="bold",fontsize=10.5)
ax.set_xlim(0,1); ax.set_ylim(0,1); ax.legend(fontsize=8,loc="upper left",frameon=False)
fig.tight_layout(); fig.savefig(OUT/f"taste-breadth{SUF}.pdf"); plt.close(fig)
print("wrote", f"taste-breadth{SUF}.pdf")
