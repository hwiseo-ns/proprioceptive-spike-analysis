#!/usr/bin/env python3
"""
make_browser2.py — two-panel interactive HTML viewer.

PANEL 1  spike pattern
    - full-recording overview strip (all 511 s) with a draggable window marker
    - detail raster for the current window: x = time, y = |amplitude|,
      colour = cluster; grey = intruder, pale grey = no burst
    - click any spike -> its cluster number is captured into slot A or B
    - window controls: start/span boxes, zoom in/out, pan, presets

PANEL 2  multichannel p2p analysis
    - same content as before (Ch1-7 median waveform with p2p annotated,
      cross-channel p2p, latency) but TWO clusters overlaid in one graph
    - slots A and B set by clicking in panel 1 or by typing a cluster number
    - difference readout: p2p ratio profile of A vs B and their template r

Same build technique as before: data serialised to JSON and pasted into the HTML,
plots drawn as hand-built inline SVG, full re-render on every state change.

  python make_browser2.py --work ./work --out spike_browser.html
"""
import argparse, json
import numpy as np

FS = 30000.0
PRE, POST = 60, 90


def build(work):
    P = np.load('./results/profiles.npz')
    C = np.load(f'./results/clusters.npz', allow_pickle=True)
    ids, prof, p2p, lat = P['cluster_ids'], P['prof'], P['p2p'], P['lat']
    peel0 = set(P['peel0_neg'].tolist())
    out, t, amp, fam = C['out'], C['t'], C['amp'], C['fam']
    meta = {int(c[2]): dict(burst=int(c[0]), fam='POS' if c[1] else 'NEG',
                            peel=int(c[3]), n=int(c[4]),
                            amp=round(float(c[5]), 1), t=round(float(c[6]), 1))
            for c in C['info']}
    # normalised templates (Ch7) so we can report r between any two clusters
    Tn = []
    for k in range(len(ids)):
        v = prof[k, -1] - prof[k, -1].mean()
        nv = np.linalg.norm(v)
        Tn.append((v / nv) if nv > 0 else v)
    Tn = np.stack(Tn)
    G = Tn @ Tn.T

    clusters = {}
    for k, cid in enumerate(ids):
        cid = int(cid)
        clusters[str(cid)] = dict(
            id=cid,
            prof=[[round(float(v), 2) for v in prof[k, ch]] for ch in range(prof.shape[1])],
            p2p=[round(float(v), 2) for v in p2p[k]],
            lat=[round(float(v), 3) for v in lat[k]],
            peel0=cid in peel0, **meta.get(cid, {}))
    # every event: time, amplitude, cluster (-1 no burst, -2 intruder), polarity
    spikes = dict(
        t=[round(float(v), 4) for v in t],
        a=[round(float(v), 1) for v in amp],
        c=[int(v) for v in out],
        f=[int(v) for v in fam])
    order = {str(int(c)): i for i, c in enumerate(ids)}
    return dict(clusters=clusters, spikes=spikes, order=order,
                gram=[[round(float(x), 4) for x in row] for row in G],
                bursts=[[round(float(a_), 3), round(float(b_), 3)] for a_, b_ in C['burst_edges']],
                dur=float(t.max()) + 1.0)


TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8">
<title>Spike pattern + multichannel browser</title>
<style>
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#faf9f7;color:#1a1a1a}
h2{font-size:14px;margin:0 0 8px;font-weight:600}
.wrap{padding:12px 18px}
.panel{background:#fff;border:1px solid #e2e0dc;border-radius:9px;padding:12px;margin-bottom:12px}
.bar{display:flex;gap:7px;align-items:center;flex-wrap:wrap;margin-bottom:8px;font-size:13px}
button{padding:5px 11px;border:1px solid #cfccc6;background:#fff;border-radius:6px;cursor:pointer;font-size:12.5px}
button:hover{background:#f0eee9}button.on{background:#1a1a1a;color:#fff;border-color:#1a1a1a}
input{width:66px;padding:4px 6px;border:1px solid #cfccc6;border-radius:5px;font-size:12.5px}
.meta{font-size:12.5px;color:#555}
.slotA{color:#1f77b4;font-weight:600}.slotB{color:#e8871a;font-weight:600}
#grid{display:grid;grid-template-columns:repeat(4,1fr);gap:9px}
.card{border:1px solid #eceae6;border-radius:7px;padding:6px}
.card h3{margin:0 0 3px;font-size:11.5px;font-weight:600}
svg{display:block;width:100%;height:auto}
table{border-collapse:collapse;font-size:12px}td,th{padding:2px 8px;border-bottom:1px solid #eee;text-align:right}
th:first-child,td:first-child{text-align:left}
</style></head><body><div class="wrap">

<div class="panel">
<h2>1 — spike pattern</h2>
<div class="bar">
  <span class="meta">overview (click or drag to move the window):</span>
</div>
<svg id="ov" viewBox="0 0 1200 60" style="cursor:crosshair"></svg>
<div class="bar" style="margin-top:10px">
  start <input id="t0" value="210"> s &nbsp; span <input id="tw" value="10"> s
  <button onclick="zoom(0.5)">zoom in</button>
  <button onclick="zoom(2)">zoom out</button>
  <button onclick="pan(-1)">&lt; pan</button>
  <button onclick="pan(1)">pan &gt;</button>
  <button onclick="setwin(0,511)">whole recording</button>
  <span class="meta">| click a spike to load it into</span>
  <button id="ta" class="on" onclick="setTarget('A')">slot A</button>
  <button id="tb" onclick="setTarget('B')">slot B</button>
  <span class="meta" id="pick"></span>
</div>
<svg id="ras" viewBox="0 0 1200 300" style="cursor:pointer"></svg>
<div class="meta" id="rasinfo"></div>
</div>

<div class="panel">
<h2>2 — multichannel p2p analysis (two clusters overlaid)</h2>
<div class="bar">
  <span class="slotA">A</span> <input id="ca" value="">
  <span class="slotB">B</span> <input id="cb" value="">
  <button onclick="applyIds()">plot</button>
  <button onclick="clearB()">clear B</button>
  <button onclick="stepSlot('A',-1)">A −</button><button onclick="stepSlot('A',1)">A +</button>
  <button onclick="stepSlot('B',-1)">B −</button><button onclick="stepSlot('B',1)">B +</button>
  <span class="meta" id="cmp"></span>
</div>
<div id="grid"></div>
<div class="bar" style="margin-top:10px"><div id="tab"></div></div>
</div>

</div><script>
const D=__DATA__, CL=D.clusters, SP=D.spikes, CH=[1,2,3,4,5,6,7];
const IDS=Object.keys(CL).map(Number).sort((x,y)=>x-y);
let t0=210, tw=10, target='A', A=null, B=null;
const COL=['#1f77b4','#e8871a'];

function hue(c){ // stable colour per cluster id
  if(c===-1) return '#d8d5d0'; if(c===-2) return '#9b968e';
  const h=(c*47)%360; return `hsl(${h},58%,45%)`;
}
function setTarget(s){target=s;document.getElementById('ta').classList.toggle('on',s=='A');
  document.getElementById('tb').classList.toggle('on',s=='B');}
function setwin(a,b){t0=a;tw=b;sync();}
function zoom(f){const c=t0+tw/2;tw=Math.max(0.2,Math.min(D.dur,tw*f));t0=Math.max(0,c-tw/2);sync();}
function pan(d){t0=Math.max(0,Math.min(D.dur-tw,t0+d*tw*0.8));sync();}
function sync(){document.getElementById('t0').value=t0.toFixed(2);
  document.getElementById('tw').value=tw.toFixed(2);drawOv();drawRas();}
['t0','tw'].forEach(id=>document.getElementById(id).addEventListener('change',()=>{
  t0=parseFloat(document.getElementById('t0').value)||0;
  tw=Math.max(0.2,parseFloat(document.getElementById('tw').value)||10);sync();}));

// ---------- overview: spike-count histogram over the whole recording
function drawOv(){
  const W=1200,H=60,pad=4,nb=400,h=new Array(nb).fill(0);
  SP.t.forEach(v=>{const k=Math.floor(v/D.dur*nb);if(k>=0&&k<nb)h[k]++;});
  const mx=Math.max(...h); let s='';
  for(let k=0;k<nb;k++){const bh=(h[k]/mx)*(H-2*pad);
    s+=`<rect x="${pad+k*(W-2*pad)/nb}" y="${H-pad-bh}" width="${(W-2*pad)/nb}" height="${bh}" fill="#b9b4ab"/>`;}
  const x1=pad+t0/D.dur*(W-2*pad), x2=pad+(t0+tw)/D.dur*(W-2*pad);
  s+=`<rect x="${x1}" y="0" width="${Math.max(2,x2-x1)}" height="${H}" fill="#1f77b4" fill-opacity="0.25" stroke="#1f77b4"/>`;
  document.getElementById('ov').innerHTML=s;
}
document.getElementById('ov').addEventListener('click',e=>{
  const r=e.currentTarget.getBoundingClientRect();
  const frac=(e.clientX-r.left)/r.width; t0=Math.max(0,Math.min(D.dur-tw,frac*D.dur-tw/2)); sync();});

// ---------- detail raster: x = time, y = amplitude, colour = cluster
let hitmap=[];
function drawRas(){
  const W=1200,H=300,pl=42,pr=8,pt=8,pb=26,amin=11,amax=42;
  const X=v=>pl+(v-t0)/tw*(W-pl-pr), Y=a=>H-pb-(a-amin)/(amax-amin)*(H-pt-pb);
  let s='',n=0,nm=0; hitmap=[];
  D.bursts.forEach(b=>{ if(b[1]>=t0&&b[0]<t0+tw)
    s+=`<rect x="${X(Math.max(b[0],t0))}" y="${pt}" width="${Math.max(1,X(Math.min(b[1],t0+tw))-X(Math.max(b[0],t0)))}" height="${H-pt-pb}" fill="#f3f1ed"/>`;});
  for(let a=15;a<=40;a+=5) s+=`<line x1="${pl}" y1="${Y(a)}" x2="${W-pr}" y2="${Y(a)}" stroke="#eee"/><text x="${pl-6}" y="${Y(a)+3}" font-size="9" text-anchor="end" fill="#888">${a}</text>`;
  for(let k=0;k<SP.t.length;k++){
    const tt=SP.t[k]; if(tt<t0||tt>t0+tw) continue; n++;
    const c=SP.c[k], x=X(tt), y=Y(Math.min(SP.a[k],amax));
    if(c>=0) nm++;
    const sel=(c>=0&&(c===A||c===B))?2.6:1.4;
    const col=(c===A)?COL[0]:(c===B)?COL[1]:hue(c);
    s+=`<line x1="${x.toFixed(1)}" y1="${y.toFixed(1)}" x2="${x.toFixed(1)}" y2="${(H-pb).toFixed(1)}" stroke="${col}" stroke-width="${sel}" opacity="${c>=0?0.95:0.55}"/>`;
    s+=`<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${c>=0?2.6:1.8}" fill="${col}"/>`;
    hitmap.push([x,y,k]);
  }
  s+=`<text x="${pl-30}" y="${pt+40}" font-size="10" fill="#666" transform="rotate(-90 ${pl-30} ${pt+40})">|amp| µV</text>`;
  for(let g=0;g<=5;g++){const tv=t0+g*tw/5;
    s+=`<text x="${X(tv)}" y="${H-8}" font-size="10" text-anchor="middle" fill="#666">${tv.toFixed(2)}</text>`;}
  document.getElementById('ras').innerHTML=s;
  document.getElementById('rasinfo').textContent=
    `${n} spikes in view, ${nm} in clusters, ${n-nm} intruder/no-burst — colour = cluster id (A blue, B orange when selected)`;
}
document.getElementById('ras').addEventListener('click',e=>{
  const r=e.currentTarget.getBoundingClientRect();
  const px=(e.clientX-r.left)/r.width*1200, py=(e.clientY-r.top)/r.height*300;
  let best=null,bd=1e9;
  hitmap.forEach(([x,y,k])=>{const d=(x-px)**2+(y-py)**2; if(d<bd){bd=d;best=k;}});
  if(best===null||bd>900) return;
  const c=SP.c[best];
  document.getElementById('pick').textContent =
    `t=${SP.t[best].toFixed(4)} s, |amp|=${SP.a[best]} µV, ` +
    (c>=0?`cluster ${c}`:(c===-2?'intruder':'no burst'));
  if(c>=0){ if(target=='A'){A=c;document.getElementById('ca').value=c;}
            else {B=c;document.getElementById('cb').value=c;} drawAll(); }
  else drawRas();
});

// ---------- panel 2
function applyIds(){
  const a=parseInt(document.getElementById('ca').value), b=parseInt(document.getElementById('cb').value);
  A=CL[a]?a:null; B=CL[b]?b:null; drawAll();
}
function clearB(){B=null;document.getElementById('cb').value='';drawAll();}
function stepSlot(s,d){
  const cur=(s=='A')?A:B; let k=IDS.indexOf(cur); if(k<0)k=0;
  const nid=IDS[(k+d+IDS.length)%IDS.length];
  if(s=='A'){A=nid;document.getElementById('ca').value=nid;} else {B=nid;document.getElementById('cb').value=nid;}
  drawAll();
}
function line(vals,w,h,pad,ymin,ymax,col,wd){
  let p=''; const n=vals.length;
  for(let k=0;k<n;k++){const x=pad+k*(w-2*pad)/(n-1),y=h-pad-(vals[k]-ymin)/(ymax-ymin)*(h-2*pad);
    p+=(k?'L':'M')+x.toFixed(1)+' '+y.toFixed(1);}
  return `<path d="${p}" fill="none" stroke="${col}" stroke-width="${wd||1.7}"/>`;
}
function drawPanel2(){
  const a=A!=null?CL[A]:null, b=B!=null?CL[B]:null;
  if(!a){document.getElementById('grid').innerHTML='<div class="meta">click a spike in panel 1, or type a cluster number</div>';
         document.getElementById('tab').innerHTML='';return;}
  let ymin=1e9,ymax=-1e9;
  [a,b].forEach(d=>{if(d)d.prof.forEach(p=>p.forEach(v=>{ymin=Math.min(ymin,v);ymax=Math.max(ymax,v)}));});
  const pad=6,rg=ymax-ymin;ymin-=rg*0.12;ymax+=rg*0.30;
  let html='';
  CH.forEach((ch,k)=>{
    const w=260,h=140;
    const zy=h-pad-(0-ymin)/(ymax-ymin)*(h-2*pad);
    let inner=`<line x1="${pad}" y1="${zy}" x2="${w-pad}" y2="${zy}" stroke="#e6e6e6"/>`;
    [[a,COL[0]],[b,COL[1]]].forEach(([d,c])=>{ if(!d) return;
      const vals=d.prof[k]; inner+=line(vals,w,h,pad,ymin,ymax,c);
      let it=0; vals.forEach((v,j)=>{if(v<vals[it])it=j;});
      const X=pad+it*(w-2*pad)/(vals.length-1);
      const Y1=h-pad-(Math.max(...vals)-ymin)/(ymax-ymin)*(h-2*pad);
      const Y2=h-pad-(Math.min(...vals)-ymin)/(ymax-ymin)*(h-2*pad);
      inner+=`<line x1="${X}" y1="${Y1}" x2="${X}" y2="${Y2}" stroke="${c}" stroke-width="1" stroke-dasharray="3 2" opacity="0.8"/>`;});
    const lbl=b?`${a.p2p[k].toFixed(1)} / ${b.p2p[k].toFixed(1)}`:`${a.p2p[k].toFixed(1)}`;
    html+=`<div class="card"><h3>Ch${ch} &nbsp; p2p ${lbl} µV</h3><svg viewBox="0 0 ${w} ${h}">${inner}</svg></div>`;
  });
  const w=260,h=140,bw=(w-2*pad)/7;
  const mx=Math.max(...a.p2p,...(b?b.p2p:[0]));
  let bars='';
  a.p2p.forEach((v,k)=>{const bh=(v/mx)*(h-2*pad-12);
    bars+=`<rect x="${pad+k*bw+2}" y="${h-pad-bh}" width="${b?(bw-6)/2:bw-4}" height="${bh}" fill="${COL[0]}"/>`;});
  if(b) b.p2p.forEach((v,k)=>{const bh=(v/mx)*(h-2*pad-12);
    bars+=`<rect x="${pad+k*bw+2+(bw-6)/2+1}" y="${h-pad-bh}" width="${(bw-6)/2}" height="${bh}" fill="${COL[1]}"/>`;});
  html+=`<div class="card"><h3>cross-channel p2p (Ch1→Ch7)</h3><svg viewBox="0 0 ${w} ${h}">${bars}</svg></div>`;
  const lm=Math.max(0.12,...a.lat.map(Math.abs),...(b?b.lat.map(Math.abs):[0])),zy2=h/2;
  let lb=`<line x1="${pad}" y1="${zy2}" x2="${w-pad}" y2="${zy2}" stroke="#c0392b"/>`;
  a.lat.forEach((v,k)=>{const bh=-(v/lm)*(h/2-pad);
    lb+=`<rect x="${pad+k*bw+2}" y="${Math.min(zy2,zy2+bh)}" width="${b?(bw-6)/2:bw-4}" height="${Math.abs(bh)||1}" fill="${COL[0]}"/>`;});
  if(b) b.lat.forEach((v,k)=>{const bh=-(v/lm)*(h/2-pad);
    lb+=`<rect x="${pad+k*bw+2+(bw-6)/2+1}" y="${Math.min(zy2,zy2+bh)}" width="${(bw-6)/2}" height="${Math.abs(bh)||1}" fill="${COL[1]}"/>`;});
  html+=`<div class="card"><h3>trough latency re Ch7 (±${lm.toFixed(2)} ms)</h3><svg viewBox="0 0 ${w} ${h}">${lb}</svg></div>`;
  const rat=a.p2p.map((v,k)=>v/a.p2p[6]);
  const ratB=b?b.p2p.map((v,k)=>v/b.p2p[6]):null;
  let rl=''; const rmax=Math.max(...rat,...(ratB||[0]))*1.1;
  rl+=line(rat.map(v=>v),w,h,pad,0,rmax,COL[0],2);
  if(ratB) rl+=line(ratB,w,h,pad,0,rmax,COL[1],2);
  html+=`<div class="card"><h3>profile: p2p / Ch7 p2p</h3><svg viewBox="0 0 ${w} ${h}">${rl}</svg></div>`;
  document.getElementById('grid').innerHTML=html;

  let rows=`<table><tr><th>property</th><th class="slotA">A ${a.id}</th>`+(b?`<th class="slotB">B ${b.id}</th><th>Δ</th>`:'')+`</tr>`;
  const add=(k,va,vb,d)=>{rows+=`<tr><td>${k}</td><td>${va}</td>`+(b?`<td>${vb}</td><td>${d}</td>`:'')+`</tr>`;};
  add('burst',a.burst,b?b.burst:'',b?Math.abs(a.burst-b.burst):'');
  add('family / peel',`${a.fam} / ${a.peel}`,b?`${b.fam} / ${b.peel}`:'','');
  add('spikes',a.n,b?b.n:'','');
  add('amp mode (µV)',a.amp,b?b.amp:'',b?(a.amp-b.amp).toFixed(1):'');
  add('time (s)',a.t,b?b.t:'',b?Math.abs(a.t-b.t).toFixed(1):'');
  add('Ch7 p2p (µV)',a.p2p[6].toFixed(1),b?b.p2p[6].toFixed(1):'',b?(a.p2p[6]-b.p2p[6]).toFixed(1):'');
  if(b){
    const ia=D.order[String(a.id)], ib=D.order[String(b.id)];
    const r=D.gram[ia][ib];
    let dp=0; for(let k=0;k<6;k++){const x=a.p2p[k]/a.p2p[6]-b.p2p[k]/b.p2p[6]; dp+=x*x;}
    add('Ch7 template r','','',r.toFixed(4));
    add('profile distance','','',Math.sqrt(dp).toFixed(4));
  }
  rows+='</table>';
  document.getElementById('cmp').textContent = b
    ? 'within-cluster split-half distance is ~0.093 — differences below that are noise'
    : 'set slot B to overlay a second cluster';
  document.getElementById('tab').innerHTML=rows;
}
function drawAll(){drawRas();drawPanel2();}
addEventListener('keydown',e=>{if(e.key=='ArrowRight')pan(1);if(e.key=='ArrowLeft')pan(-1);});
A=IDS[0];document.getElementById('ca').value=A;
sync();drawPanel2();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', default='./work')
    ap.add_argument('--out', default='./results/spike_browser.html')
    a = ap.parse_args()
    D = build(a.work)
    html = TEMPLATE.replace('__DATA__', json.dumps(D))
    open(a.out, 'w', encoding='utf-8').write(html)
    print(f'wrote {a.out} ({len(html)/1e6:.1f} MB, {len(D["clusters"])} clusters, '
          f'{len(D["spikes"]["t"])} spikes)')


if __name__ == '__main__':
    main()