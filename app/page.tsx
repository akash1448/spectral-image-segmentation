'use client';

import { useEffect, useMemo, useRef, useState } from 'react';


type Matrix = number[][];
type Results = {
  width:number; height:number; pixels:number; edges:number; density:number;
  adjacency: Matrix; laplacian: Matrix; eigenvalues:number[]; eigenvectors: Matrix; fiedler:number[]; clusters:number[]; clusterImages:string[]; original:string; resized:string;
};

const clamp=(n:number,a:number,b:number)=>Math.max(a,Math.min(b,n));

function jacobiEigen(Ain: Matrix){
  const n=Ain.length; const A=Ain.map(r=>r.slice()); const V=Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>i===j?1:0));
  const maxIter=Math.min(120,n*n*3);
  for(let iter=0;iter<maxIter;iter++){
    let p=0,q=1,max=0;
    for(let i=0;i<n;i++) for(let j=i+1;j<n;j++){ const v=Math.abs(A[i][j]); if(v>max){max=v;p=i;q=j;} }
    if(max<1e-7) break;
    const phi=0.5*Math.atan2(2*A[p][q],A[p][p]-A[q][q]); const c=Math.cos(phi),s=Math.sin(phi);
    const app=c*c*A[p][p]-2*s*c*A[p][q]+s*s*A[q][q]; const aqq=s*s*A[p][p]+2*s*c*A[p][q]+c*c*A[q][q];
    for(let k=0;k<n;k++){ if(k!==p&&k!==q){const akp=A[k][p],akq=A[k][q]; A[k][p]=A[p][k]=c*akp-s*akq; A[k][q]=A[q][k]=s*akp+c*akq;} }
    A[p][p]=app; A[q][q]=aqq; A[p][q]=A[q][p]=0;
    for(let k=0;k<n;k++){const vkp=V[k][p],vkq=V[k][q]; V[k][p]=c*vkp-s*vkq; V[k][q]=s*vkp+c*vkq;}
  }
  const pairs=A.map((r,i)=>({val:r[i],vec:V.map(x=>x[i])})).sort((a,b)=>a.val-b.val);
  return pairs;
}

function buildGraph(rgb:Float32Array,w:number,h:number,sigma:number,neigh:4|8){
  const n=w*h; const A=Array.from({length:n},()=>Array(n).fill(0)); let edges=0;
  const dirs=neigh===4?[[1,0],[-1,0],[0,1],[0,-1]]:[[1,0],[-1,0],[0,1],[0,-1],[1,1],[-1,-1],[1,-1],[-1,1]];
  const idx=(x:number,y:number)=>y*w+x;
  for(let y=0;y<h;y++) for(let x=0;x<w;x++){
    const i=idx(x,y); for(const [dx,dy] of dirs){const xx=x+dx,yy=y+dy;if(xx<0||xx>=w||yy<0||yy>=h)continue; const j=idx(xx,yy); if(A[i][j]>0)continue;
      const dr=rgb[3*i]-rgb[3*j],dg=rgb[3*i+1]-rgb[3*j+1],db=rgb[3*i+2]-rgb[3*j+2]; const d2=dr*dr+dg*dg+db*db;
      const wgt=Math.exp(-d2/(2*sigma*sigma)); A[i][j]=wgt; A[j][i]=wgt; edges++;
    }
  }
  const L=Array.from({length:n},()=>Array(n).fill(0)); for(let i=0;i<n;i++){let d=0;for(let j=0;j<n;j++)d+=A[i][j];L[i][i]=d;for(let j=0;j<n;j++)if(i!==j)L[i][j]=-A[i][j];}
  return {A,L,edges};
}

function colorizeClusters(colors:Uint8ClampedArray, labels:number[], w:number,h:number,k:number, alpha=false){
  const palette=[[42,167,255],[255,99,132],[116,87,255],[0,210,154],[255,181,71]];
  const out=new Uint8ClampedArray(w*h*4);
  for(let i=0;i<labels.length;i++){const p=palette[labels[i]%palette.length]; if(alpha){out[4*i]=p[0];out[4*i+1]=p[1];out[4*i+2]=p[2];out[4*i+3]=220;} else {const mix=0.62; out[4*i]=Math.round(colors[3*i]*(1-mix)+p[0]*mix);out[4*i+1]=Math.round(colors[3*i+1]*(1-mix)+p[1]*mix);out[4*i+2]=Math.round(colors[3*i+2]*(1-mix)+p[2]*mix);out[4*i+3]=255;}}
  return out;
}
function makeURL(data:Uint8ClampedArray,w:number,h:number){const c=document.createElement('canvas');c.width=w;c.height=h;const ctx=c.getContext('2d')!;ctx.putImageData(new ImageData(data,w,h),0,0);return c.toDataURL('image/png');}
function quantileScale(v:number[]){const min=Math.min(...v),max=Math.max(...v);return v.map(x=>(x-min)/(max-min||1));}

async function processImage(src:string, resolution:number, clusters:number, neighborhood:4|8, sigma:number):Promise<Results>{
  const img=new Image(); img.src=src; await img.decode();
  const maxSide=resolution; const ratio=Math.min(1,maxSide/Math.max(img.naturalWidth,img.naturalHeight)); const w=Math.max(2,Math.round(img.naturalWidth*ratio)),h=Math.max(2,Math.round(img.naturalHeight*ratio));
  const c=document.createElement('canvas');c.width=w;c.height=h;const ctx=c.getContext('2d',{willReadFrequently:true})!;ctx.drawImage(img,0,0,w,h);const id=ctx.getImageData(0,0,w,h);const rgb=new Float32Array(w*h*3);for(let i=0;i<w*h;i++){rgb[3*i]=id.data[4*i]/255*255;rgb[3*i+1]=id.data[4*i+1]/255*255;rgb[3*i+2]=id.data[4*i+2]/255*255;}
  const originalData=new Uint8ClampedArray(w*h*3); originalData.set(rgb as any);
  const {A,L,edges}=buildGraph(rgb,w,h,sigma,neighborhood); const eig=jacobiEigen(L); const n=w*h; const take=Math.min(clusters,n-1); const eigenvalues=eig.slice(0,Math.min(8,eig.length)).map(x=>Math.max(0,x.val));
  const eigenvectors=eig.slice(0,take+1).map(x=>x.vec); const fiedler=eig[1]?.vec || eig[0].vec;
  // Sign split is exact for 2 clusters; for k>2, use the first k eigenvectors as an embedding and k-means.
  const embedding=Array.from({length:w*h},(_,i)=>eig.slice(1,take+1).map(e=>e.vec[i]));
  let labels:number[]=[];
  if(clusters===2) labels=fiedler.map(v=>v>=0?0:1);
  else {
    let cents=Array.from({length:clusters},(_,k)=>embedding[Math.floor((w*h-1)*k/Math.max(1,clusters-1))].slice());
    for(let it=0;it<16;it++){labels=embedding.map(p=>{let bi=0,bd=Infinity;cents.forEach((c,j)=>{let d=0;for(let q=0;q<p.length;q++)d+=(p[q]-c[q])**2;if(d<bd){bd=d;bi=j;}});return bi;}); const sums=Array.from({length:clusters},()=>Array(take).fill(0));const cnt=Array(clusters).fill(0); embedding.forEach((p,i)=>{const l=labels[i];cnt[l]++;p.forEach((v,q)=>sums[l][q]+=v);});cents=cents.map((c,j)=>cnt[j]?sums[j].map(v=>v/cnt[j]):c);}
  }
  const seg= colorizeClusters(rgb as any,labels,w,h,clusters,false); const mask= colorizeClusters(rgb as any,labels,w,h,clusters,true);
  return {width:w,height:h,pixels:w*h,edges,density:edges/(w*h),adjacency:A,laplacian:L,eigenvalues,eigenvectors:eigenvectors as Matrix,fiedler,clusters:labels,clusterImages:[makeURL(seg,w,h),makeURL(mask,w,h)],original:src,resized:makeURL(id.data,w,h)};
}

function heatMatrix(m:Matrix, size:number){const n=m.length;const step=Math.max(1,Math.floor(n/size));const out=Array.from({length:Math.ceil(n/step)},()=>Array(Math.ceil(n/step)).fill(0));for(let i=0;i<out.length;i++)for(let j=0;j<out.length;j++){out[i][j]=m[Math.min(n-1,i*step)][Math.min(n-1,j*step)]}return out;}

export default function Home(){
  const [src,setSrc]=useState(''); const [clusters,setClusters]=useState(2); const [neigh,setNeigh]=useState<4|8>(8); const [sigma,setSigma]=useState(35); const [resolution,setResolution]=useState(55); const [tab,setTab]=useState('Overview'); const [res,setRes]=useState<Results|null>(null); const [busy,setBusy]=useState(false); const inputRef=useRef<HTMLInputElement>(null);
  const run=async()=>{if(!src)return;setBusy(true);try{setRes(await processImage(src,resolution,clusters,neigh,sigma));}finally{setBusy(false)}};
  useEffect(()=>{if(src)run();/* eslint-disable-next-line react-hooks/exhaustive-deps */},[clusters,neigh,sigma,resolution]);
  const fScale=useMemo(()=>res?quantileScale(res.fiedler):[],[res]);
  const matrixA=useMemo(()=>res?heatMatrix(res.adjacency,38):[],[res]); const matrixL=useMemo(()=>res?heatMatrix(res.laplacian,38):[],[res]);
  const upload=async(e:React.ChangeEvent<HTMLInputElement>)=>{const f=e.target.files?.[0];if(!f)return;setSrc(URL.createObjectURL(f));};

  const today = new Date().toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' });

  return <main>
    <header className="masthead">
      <div className="hero-inner" style={{paddingBottom:0,paddingTop:12}}>
        <div className="edition-bar">
          <span>Vol. 1 · Linear Algebra Project</span>
          <span className="red">Spectral Methods Edition</span>
          <span>{today}</span>
        </div>
      </div>
      <div className="hero-inner">
        <div className="badge">Breaking · Eigenvectors</div>
        <h1>Spectral Image<br/><span className="red">Segmentation</span><br/>Using Eigenvectors</h1>
        <p className="hero-lead">Turn an image into a graph, extract the Laplacian&apos;s eigenstructure, and watch the Fiedler vector separate visual regions — the same linear algebra that finds communities in social networks.</p>
        <div className="hero-actions">
          <button className="btn-primary" onClick={()=>inputRef.current?.click()}><span>↑</span> Upload an image</button>
          <button className="btn-secondary" onClick={()=>document.getElementById('how')?.scrollIntoView({behavior:'smooth'})}>How it works <span>↓</span></button>
        </div>
        <input ref={inputRef} hidden type="file" accept="image/*" onChange={upload}/>
      </div>
    </header>

    <nav className="tabs">{['Overview','Matrices','Eigenvalues','Segments','Connection'].map(t=><button key={t} className={tab===t?'active':''} onClick={()=>{setTab(t);document.getElementById(t==='Connection'?'connection': 'workspace')?.scrollIntoView({behavior:'smooth'})}}>{t}</button>)}</nav>

    <section className="workspace newsprint-texture" id="workspace">
      <aside className="sidebar">
        <div className="side-title">Parameters</div>
        <label>Number of clusters <b>{clusters}</b></label>
        <input type="range" min="2" max="5" value={clusters} onChange={e=>setClusters(+e.target.value)}/>
        <div className="range-tags"><span>2</span><span>3</span><span>4</span><span>5</span></div>

        <label>Neighbourhood</label>
        <div className="seg-control">
          <button className={neigh===4?'on':''} onClick={()=>setNeigh(4)}>4-neighbour</button>
          <button className={neigh===8?'on':''} onClick={()=>setNeigh(8)}>8-neighbour</button>
        </div>

        <label>Similarity σ <b>{sigma}</b></label>
        <input type="range" min="5" max="80" step="5" value={sigma} onChange={e=>setSigma(+e.target.value)}/>

        <label>Computation resolution <b>{resolution}px</b></label>
        <select value={resolution} onChange={e=>setResolution(+e.target.value)}>
          <option value="45">45 px</option>
          <option value="55">55 px</option>
          <option value="70">70 px</option>
          <option value="90">90 px</option>
        </select>

        <div className="tip"><span>ⓘ</span><span>Only nearby pixels are connected, keeping A sparse and the eigenproblem practical.</span></div>
      </aside>

      <div className="maincol">
        <div className="section-head">
          <div>
            <div className="eyebrow">Image Input · Fig. 1</div>
            <h2>From pixels to a graph</h2>
          </div>
          {busy && <span className="processing"><span className="spinner"/> Computing eigensystem…</span>}
        </div>

        {!src ? (
          <div className="drop hard-shadow-hover" onClick={()=>inputRef.current?.click()}>
            <div className="drop-icon">▧</div>
            <h3>Drop an image here</h3>
            <p>or click to browse · PNG / JPG / WEBP</p>
          </div>
        ) : (
          <div className="image-grid">
            <div className="image-card"><img src={src} alt="Original upload"/><span>Original image</span></div>
            <div className="image-card"><img src={res?.resized || src} alt="Resized for computation"/><span>Resized for graph · {res?`${res.width}×${res.height}`:`≤ ${resolution}px`}</span></div>
          </div>
        )}

        <div className="pipeline" id="how">
          <div className="pipeline-title">Mathematical Pipeline</div>
          <div className="pipe-row">
            {[['Image','Pixels as graph nodes'],['Similarity','Matrix A'],['Degree','Matrix D'],['Laplacian','L = D − A'],['Eigen','λ, v'],['Fiedler','2nd-smallest'],['Cluster','Regions']].map((x,i)=>(
              <div className="pipe-step" key={x[0]}>
                <strong>{x[0]}</strong>
                <span>{x[1]}</span>
                {i<6 && <i>→</i>}
              </div>
            ))}
          </div>
        </div>

        {res && <>
          <div className="stats">
            <div><b>{res.pixels.toLocaleString()}</b><span>graph nodes</span></div>
            <div><b>{res.edges.toLocaleString()}</b><span>undirected edges</span></div>
            <div><b>{(res.edges/(res.pixels||1)).toFixed(1)}</b><span>avg. connections</span></div>
            <div><b>{res.eigenvalues[0].toFixed(4)}</b><span>smallest λ</span></div>
          </div>

          <section className="two-col">
            <div className="pane">
              <div className="pane-head">
                <div><span className="eyebrow">Matrix A</span><h3>Similarity / adjacency</h3></div>
                <span className="font-mono" style={{fontSize:12}}>▦</span>
              </div>
              <Heatmap matrix={matrixA}/>
              <p>Each pixel is connected to nearby pixels. Similar colour → larger weight.</p>
            </div>
            <div className="pane">
              <div className="pane-head">
                <div><span className="eyebrow">Matrix L</span><h3>Graph Laplacian</h3></div>
                <span className="font-mono" style={{fontSize:12}}>⌁</span>
              </div>
              <Heatmap matrix={matrixL}/>
              <p>We build <b>L = D − A</b>, where D stores each node&apos;s total connection weight.</p>
            </div>
          </section>

          <section className="eigen-card">
            <div className="pane-head">
              <div><span className="eyebrow">Eigenvalue Decomposition</span><h3>Smallest eigenvalues of L</h3></div>
              <span className="font-mono">λ</span>
            </div>
            <div className="eigen-row">
              {res.eigenvalues.slice(0,Math.min(6,res.eigenvalues.length)).map((v,i)=>(
                <div className={i===1?'eig selected':'eig'} key={i}>
                  <span>λ{i}</span>
                  <strong>{v.toFixed(5)}</strong>
                  {i===1 && <em>Fiedler</em>}
                </div>
              ))}
            </div>
            <div className="callout">
              <span>λ₂ →</span>
              <div>
                <b>The second-smallest eigenvalue reveals the first non-trivial split.</b>
                <p>The eigenvector paired with λ₂ is the <b>Fiedler vector</b>. Its sign separates the graph into two groups.</p>
              </div>
            </div>
          </section>

          <section className="two-col">
            <div className="pane">
              <div className="pane-head">
                <div><span className="eyebrow">Fiedler Vector</span><h3>Signed pixel field</h3></div>
                <span className="font-mono">Σ</span>
              </div>
              <div className="field">
                <canvas ref={c=>{if(c&&res){c.width=res.width;c.height=res.height;const ctx=c.getContext('2d')!;const sc=quantileScale(res.fiedler);const id=ctx.createImageData(res.width,res.height);for(let i=0;i<sc.length;i++){const z=sc[i];const r=Math.round(255*Math.max(0,(.5-z)*2));const b=Math.round(255*Math.max(0,(z-.5)*2));const g=Math.round(220*(1-Math.abs(z-.5)*2));id.data[4*i]=r;id.data[4*i+1]=g;id.data[4*i+2]=b;id.data[4*i+3]=255;}ctx.putImageData(id,0,0)}}}/>
              </div>
              <div className="legend">
                <span><i className="neg"/> negative</span>
                <span><i className="pos"/> positive</span>
              </div>
              <div className="hist">
                {Array.from({length:30}).map((_,i)=>{const n=fScale.filter(v=>Math.floor(v*30)===i).length;return <span key={i} style={{height:`${8+n*2}px`}}/>})}
              </div>
            </div>
            <div className="pane">
              <div className="pane-head">
                <div><span className="eyebrow">Segmentation</span><h3>{clusters}-cluster result</h3></div>
                <span className="font-mono">▥</span>
              </div>
              <img className="seg-img" src={res.clusterImages[0]} alt="Segmented image"/>
              <div className="cluster-list">
                {Array.from({length:clusters}).map((_,i)=>(
                  <div key={i}>
                    <span className="dot" style={{background:['#2aa7ff','#ff6384','#7457ff','#00d29a','#ffb547'][i]}}/>
                    Region {i+1}
                    <b>{Math.round(res.clusters.filter(x=>x===i).length/res.pixels*100)}%</b>
                  </div>
                ))}
              </div>
            </div>
          </section>
        </>}
      </div>
    </section>

    <div className="ornament">✦ ✦ ✦</div>

    <section className="explain">
      <div className="eyebrow">How It Works</div>
      <h2>The same spectral idea, a new domain.</h2>
      <div className="explain-grid">
        <div>
          <span>01</span>
          <h3>Pixels become nodes</h3>
          <p>An image is a grid of pixels. We treat every pixel as a graph node and link it to nearby pixels.</p>
        </div>
        <div>
          <span>02</span>
          <h3>Similarity becomes edge weight</h3>
          <p>For neighbours i and j, <b>w(i,j)=exp(−‖xᵢ−xⱼ‖² / 2σ²)</b>. Similar colours create stronger edges.</p>
        </div>
        <div>
          <span>03</span>
          <h3>The Laplacian encodes structure</h3>
          <p>With <b>L=D−A</b>, the matrix summarizes how strongly each node is tied to its local region.</p>
        </div>
        <div>
          <span>04</span>
          <h3>Eigenvectors reveal groups</h3>
          <p>The Fiedler vector is the eigenvector for the second-smallest eigenvalue. Its sign gives a natural 2-way partition.</p>
        </div>
      </div>
    </section>

    <section className="connection" id="connection">
      <div className="eyebrow">Connection to Project 13</div>
      <h2>Social communities → image regions</h2>
      <div className="mapping">
        <div>
          <b>Project 13</b>
          <span>People</span>
          <span>Friendships</span>
          <span>Social adjacency matrix A</span>
          <span>Social Laplacian L</span>
          <span>Fiedler vector</span>
          <strong>Communities</strong>
        </div>
        <div className="arrow">⟷</div>
        <div>
          <b>This project</b>
          <span>Pixels</span>
          <span>Pixel similarity</span>
          <span>Image similarity matrix A</span>
          <span>Image Laplacian L</span>
          <span>Fiedler vector</span>
          <strong>Image regions</strong>
        </div>
      </div>
      <p className="connection-copy">The application is deliberately not a generic image-processing pipeline: the central operation is still spectral clustering through the eigenvalues and eigenvectors of the graph Laplacian — exactly the same linear-algebra idea, transferred from social networks to pixels.</p>
    </section>

    <footer>
      <span>Linear Algebra Project · Newsprint Edition</span>
      <span>Image → Graph → Matrices → Eigenvectors → Clusters</span>
    </footer>
  </main>
}

function Heatmap({matrix}:{matrix:Matrix}){
  return (
    <div className="heatmap" style={{gridTemplateColumns:`repeat(${matrix.length},1fr)`}}>
      {matrix.flat().map((v,i)=>{
        const z=Math.max(0,Math.min(1,Math.abs(v)/(Math.max(...matrix.flat().map(Math.abs))||1)));
        return <i key={i} style={{opacity:.15+.85*z}}/>
      })}
    </div>
  );
}
