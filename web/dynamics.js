/* V0.4 research observatory. No synthetic model states or heatmap interpolation. */
(() => {
  'use strict';
  const $ = s => document.querySelector(s);
  const R={run:null,manifest:null,layer:1,stepA:0,stepB:0,channel:0,sample:0,version:0,probeVersion:0,cluster:null};
  const fmt=(v,d=3)=>Number.isFinite(v)?Number(v).toFixed(d):'—';
  const box=(id,v)=>{const el=$(id);if(el)el.textContent=v};
  const failure=e=>box('#dyn-message', 'Unavailable: '+e.message);
  async function get(path){const r=await fetch(path);if(!r.ok)throw new Error(`${path.split('?')[0]}: ${r.status} ${await r.text()}`);return r.json()}
  function options(select,arr,fn){select.replaceChildren();for(const item of arr){const o=document.createElement('option');o.value=fn(item);o.textContent=fn(item);select.append(o)}}
  function url(path,params){return path+'?'+new URLSearchParams(params)}
  function drawHistogram(el,data,tint='#7884e9'){
    if(!data || !data.counts || !data.counts.length){el.replaceChildren();return}
    const counts=data.counts,max=Math.max(...counts,1),w=480,h=160;
    const rows=counts.map((v,i)=>{const bw=w/counts.length;const height=(v/max)*(h-24);return `<rect x="${(i*bw+1).toFixed(2)}" y="${(h-height).toFixed(2)}" width="${Math.max(1,bw-2).toFixed(2)}" height="${height.toFixed(2)}" rx="1.5" fill="${tint}" opacity="${.4+.6*v/max}"/>`}).join('');
    el.innerHTML=`<line x1="0" y1="${h}" x2="${w}" y2="${h}" stroke="#bbc4dc"/>${rows}`;
  }
  function drawResponse(channel){
    const svg=$('#dyn-class-response');
    if(!channel){svg.replaceChildren();return}
    const values=channel.response_by_class,max=Math.max(...values.filter(v=>v!==null),0.001),bottom=130;
    const bar=values.map((v,i)=>{if(v===null)return '';const x=18+i*46,ht=Math.max(0,v/max)*92;
      return `<rect x="${x}" y="${bottom-ht}" width="27" height="${ht}" rx="5" fill="#7585de" opacity="${.4+.6*v/max}"/><text x="${x+13}" y="${bottom+17}" font-size="12" fill="#7c88a4" text-anchor="middle">${i}</text>`}).join('');
    svg.innerHTML=`<line x1="9" y1="${bottom}" x2="475" y2="${bottom}" stroke="#dbe0ee"/>${bar}`;
    box('#dyn-channel-label','CH '+String(channel.channel).padStart(2,'0')+' · '+fmt(channel.mean_response,4));
  }
  function renderClusters(doc){
    R.cluster=doc;
    const counts=$('#dyn-cluster-counts');counts.replaceChildren();
    for(let i=0;i<doc.groups;i++){const lbl=document.createElement('span');lbl.textContent=`GROUP ${i+1} · ${doc.counts[i]}`;lbl.style.setProperty('--cluster',`var(--dyn-cluster-${i%6})`);counts.append(lbl)}
    const grid=$('#dyn-cluster-channels');grid.replaceChildren();
    for(const ch of doc.channels){const b=document.createElement('button');b.type='button';b.textContent=String(ch.channel).padStart(2,'0');b.className='dyn-channel'+(ch.channel===R.channel?' selected':'');b.style.setProperty('--cluster',`var(--dyn-cluster-${ch.cluster%6})`);
      b.title=`Channel ${ch.channel}: group ${ch.cluster+1}; mean ${fmt(ch.mean_response,4)}`;
      b.addEventListener('click',()=>{R.channel=ch.channel;renderClusters(doc);loadProbe()});grid.append(b)}
    drawResponse(doc.channels[R.channel]??doc.channels[0]);
  }
  function renderCKA(doc){
    const grid=$('#dyn-cka-matrix');grid.replaceChildren();grid.style.setProperty('--dimension',doc.steps.length);
    const steps=doc.steps;
    for(let i=0;i<steps.length;i++)for(let j=0;j<steps.length;j++){
      const v=doc.matrix[i][j],b=document.createElement('button');b.type='button';b.className='dyn-cka-cell'+(steps[i]===R.stepA&&steps[j]===R.stepB?' active':'');
      b.style.background=`hsl(${225+36*v}, ${24+47*v}%, ${96-32*v}%)`;
      b.textContent=v>=.9995?'1':fmt(v,2);
      b.setAttribute('aria-label',`Step ${steps[i]} 与 Step ${steps[j]} 表征相似度 ${fmt(v,3)}`);
      b.title=`STEP ${steps[i]} → ${steps[j]} · CKA ${fmt(v,5)}`;
      b.addEventListener('click',()=>{
        R.stepA=steps[i];R.stepB=steps[j];$('#dyn-step-a').value=R.stepA;$('#dyn-step-b').value=R.stepB;
        box('#dyn-cka-score',fmt(v,3));renderCKA(doc);loadSnapshot();loadProbe();
      });grid.append(b)
    }
    const ia=steps.indexOf(R.stepA),ib=steps.indexOf(R.stepB);
    box('#dyn-cka-score',ia>=0&&ib>=0?fmt(doc.matrix[ia][ib],3):'—');
    box('#dyn-pair-label',`A STEP ${R.stepA} → B STEP ${R.stepB}`);
  }
  let lastCKA=null;
  async function loadLayer(){
    if(!R.manifest)return;
    const v=++R.version;
    box('#dyn-message','Computing real checkpoint representations…');
    try{
      const [cka,diagnostics,clusters]=await Promise.all([
        get(url('/api/dynamics/cka',{run_id:R.run,layer:R.layer})),
        get(url('/api/dynamics/snapshot',{run_id:R.run,layer:R.layer,step:R.stepB})),
        get(url('/api/dynamics/clusters',{run_id:R.run,layer:R.layer,step:R.stepB,groups:4}))
      ]);
      if(v!==R.version)return;
      lastCKA=cka;renderCKA(cka);renderSnapshot(diagnostics);R.channel=Math.min(R.channel,clusters.channels.length-1);
      renderClusters(clusters);box('#dyn-message','Real checkpoints · Fixed held-out probes · No interpolation');loadProbe();
    }catch(e){if(v===R.version)failure(e)}
  }
  function renderSnapshot(doc){
    drawHistogram($('#dyn-weight-hist'),doc.weight_histogram);
    drawHistogram($('#dyn-grad-hist'),doc.gradient_histogram,'#d5a77b');
    box('#dyn-weight-norm',fmt(doc.weight_l2,4));box('#dyn-grad-norm',fmt(doc.gradient_preclip,4));
    box('#dyn-update-norm',fmt(doc.update_l2,5));
  }
  async function loadSnapshot(){
    if(!R.manifest)return;
    const v=++R.version;
    try{
      const [diagnostics,clusters]=await Promise.all([
        get(url('/api/dynamics/snapshot',{run_id:R.run,layer:R.layer,step:R.stepB})),
        get(url('/api/dynamics/clusters',{run_id:R.run,layer:R.layer,step:R.stepB,groups:4}))]);
      if(v!==R.version)return;
      renderSnapshot(diagnostics);R.channel=Math.min(R.channel,clusters.channels.length-1);
      renderClusters(clusters)
    }catch(e){if(v===R.version)failure(e)}
  }
  function clearImages(){for(const s of ['#dyn-input','#dyn-image-a','#dyn-image-b','#dyn-image-diff'])$(s).removeAttribute('src')}
  async function loadProbe(){
    if(!R.manifest)return;
    const v=++R.probeVersion;clearImages();
    try{
      const data=await get(url('/api/dynamics/probe',{run_id:R.run,layer:R.layer,
        step_a:R.stepA,step_b:R.stepB,channel:R.channel,sample_index:R.sample}));
      if(v!==R.probeVersion)return;
      $('#dyn-input').src=data.input_image;$('#dyn-image-a').src=data.activation_a;
      $('#dyn-image-b').src=data.activation_b;$('#dyn-image-diff').src=data.difference;
      box('#dyn-pair-cka',fmt(data.cka,3));box('#dyn-pair-rms',fmt(data.rms_difference,5));
      const delta=data.true_probability_b-data.true_probability_a;
      box('#dyn-prob',(delta>=0?'+':'')+fmt(delta*100,2)+' pp');
      box('#dyn-pred',`${data.prediction_a} → ${data.prediction_b}`);
      box('#dyn-explanation',`Fixed held-out sample ${data.sample_index}; true class ${data.label}. Absolute fixed color scale; signed difference B−A. Predictive change is observed, not established as causal.`)
    }catch(e){if(v===R.probeVersion)failure(e)}
  }
  async function loadRun(){
    const name=$('#dyn-run').value;R.run=name;R.version++;R.probeVersion++;
    try{
      const manifest=await get(url('/api/dynamics/manifest',{run_id:name}));
      if(R.run!==name)return;
      R.manifest=manifest;R.stepA=manifest.snapshots[0].step;R.stepB=manifest.snapshots.at(-1).step;
      R.channel=0;R.sample=0;R.layer=1;$('#dyn-layer').value='1';
      const steps=manifest.snapshots.map(s=>s.step);
      options($('#dyn-step-a'),steps,String);options($('#dyn-step-b'),steps,String);
      $('#dyn-step-a').value=R.stepA;$('#dyn-step-b').value=R.stepB;
      $('#dyn-sample').max=manifest.probe_indices.length-1;$('#dyn-sample').value=0;
      box('#dyn-sample-label','00');box('#dyn-arch',`${manifest.variant} · ${manifest.dataset} · ${manifest.events.length} updates`);
      box('#dyn-provenance',manifest.provenance.interpolated?'8×8 SOURCE · INTERPOLATED TO 32×32':'NATIVE 32×32 SOURCE · OFFICIAL TEST SPLIT');
      loadLayer();
    }catch(e){failure(e)}
  }
  async function boot(){
    $('#dyn-layer').addEventListener('change',e=>{R.layer=+e.target.value;R.channel=0;loadLayer()});
    $('#dyn-run').addEventListener('change',loadRun);
    $('#dyn-step-a').addEventListener('change',e=>{R.stepA=+e.target.value;if(lastCKA)renderCKA(lastCKA);loadProbe()});
    $('#dyn-step-b').addEventListener('change',e=>{R.stepB=+e.target.value;if(lastCKA)renderCKA(lastCKA);loadSnapshot();loadProbe()});
    $('#dyn-sample').addEventListener('input',e=>{R.sample=+e.target.value;box('#dyn-sample-label',String(R.sample).padStart(2,'0'));loadProbe()});
    try{
      const data=await get('/api/dynamics/runs');box('#dyn-run-count',data.runs.length);
      const select=$('#dyn-run');select.replaceChildren();
      for(const run of data.runs){const o=document.createElement('option');o.value=run.run_id;o.textContent=`${run.run_id} · ${run.variant}`;select.append(o)}
      if(!data.runs.length){box('#dyn-message','Run `python -m cnnviz.research` to create a genuine 5-layer training experiment.');return}
      await loadRun()
    }catch(e){failure(e)}
  }
  boot();
})();