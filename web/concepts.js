/* All V0.5 plots use real returned measurements; comparisons require probe alignment. */
(()=>{'use strict';
const $=id=>document.getElementById(id);
const state={runs:[],run:'',other:'',step:0,otherStep:0,layer:5,channel:0,version:0};
const f=(v,n=3)=>Number.isFinite(v)?Number(v).toFixed(n):'—';
const text=(id,v)=>{if($(id))$(id).textContent=v};
const query=(endpoint,params)=>endpoint+'?'+new URLSearchParams(params);
const get=async(url)=>{const response=await fetch(url);if(!response.ok){let err;try{err=await response.json()}catch{}throw Error(err?.detail||'HTTP '+response.status)}return response.json()};
function setOptions(id,values,selector=x=>x){const el=$(id);el.replaceChildren();for(const item of values){const option=document.createElement('option');option.value=String(selector(item));option.textContent=String(selector(item));el.append(option)}}
function notify(msg){text('cx-status',msg)}
const active=()=>({run_id:state.run,step:state.step,layer:state.layer,channel:state.channel});
async function loadRuns(){try{const data=await get('/api/dynamics/runs');state.runs=data.runs.filter(x=>x.snapshots>0);
if(!state.runs.length){notify('No saved research runs. Train with python -m cnnviz.research first.');return}
setOptions('cx-run',state.runs,x=>x.run_id);setOptions('cx-other',state.runs,x=>x.run_id);
state.run=(state.runs.find(x=>x.run_id.startsWith('v05-'))||state.runs[0]).run_id;state.other=(state.runs.find(x=>x.run_id.startsWith('v05-')&&x.run_id!==state.run)||state.runs.find(x=>x.run_id!==state.run)||state.runs[0]).run_id;$('cx-run').value=state.run;$('cx-other').value=state.other;
await selectRun();}catch(err){notify('Cannot read research runs: '+err.message)}}
async function meta(name){return get(query('/api/dynamics/manifest',{run_id:name}))}
async function selectRun(){const version=++state.version;state.run=$('cx-run').value;try{const m=await meta(state.run);if(version!==state.version)return;
const steps=m.snapshots.map(s=>s.step);setOptions('cx-step',steps);state.step=steps.at(-1);$('cx-step').value=state.step;
setOptions('cx-target',m.classes.map((_,i)=>i));$('cx-sample').max=String(m.probe_indices.length-1);
await selectLayer();await selectOther();}catch(err){notify(err.message)}}
async function selectLayer(){state.layer=Number($('cx-layer').value);state.channel=0;
const m=await meta(state.run);const channels=[24,32,48,48,64][state.layer-1];setOptions('cx-channel',Array.from({length:channels},(_,i)=>i));$('cx-channel').value='0';
await loadTopk();}
async function loadTopk(){const v=++state.version;state.step=Number($('cx-step').value);state.channel=Number($('cx-channel').value);
$('cx-gallery').replaceChildren();text('cx-purity','—');notify('Finding strongest responses in the actual held-out images…');
try{const d=await get(query('/api/concepts/topk',active()));if(v!==state.version)return;
for(const item of d.top_k){const card=document.createElement('article');card.className='cx-sample-card';
const pair=document.createElement('div');pair.className='cx-pictures';for(const [src,alt] of [[item.input,'Original held-out image'],[item.activation,'Measured feature activation']]){const img=document.createElement('img');img.src=src;img.alt=alt;pair.append(img)}
const title=document.createElement('strong');title.textContent=`#${item.probe_index} · class ${item.label}`;
const sub=document.createElement('small');sub.textContent=`peak ${f(item.peak_response,4)} · mean ${f(item.mean_response,4)}`;
card.append(pair,title,sub);$('cx-gallery').append(card)}
text('cx-purity',f(d.top_k_purity*100,0)+'%');text('cx-gallery-note',`Top-k dominant label ${d.dominant_label}: ${d.counts_in_top_k[d.dominant_label]} / ${d.top_k.length}; probe prior ${f(d.pool_prior*100,1)}%. Fixed activation scale [0, ${f(d.scale.high,4)}]. Candidate response patterns only.`);
notify(`${d.pool_size} real held-out images · ${d.probe_fingerprint} · no synthesized concept labels`);
}catch(err){if(v===state.version)notify('Top-k unavailable: '+err.message)}}
async function selectOther(){state.other=$('cx-other').value;const m=await meta(state.other);const steps=m.snapshots.map(s=>s.step);setOptions('cx-other-step',steps);state.otherStep=steps.at(-1);$('cx-other-step').value=state.otherStep}
async function compare(){const v=state.version;notify('Aligning probe images and computing representation statistics…');
try{const doc=await get(query('/api/concepts/compare',{run_a:state.run,run_b:state.other,step_a:state.step,step_b:state.otherStep,layer:state.layer}));
if(v!==state.version)return;
text('cx-cka',f(doc.cka));text('cx-ci',doc.bootstrap_ci95.map(z=>f(z)).join(' – '));text('cx-corr',f(doc.channel_mean_corr));
$('cx-matches').replaceChildren();for(const m of doc.matches){const row=document.createElement('button');row.type='button';row.className='cx-match-row';const title=document.createElement('b');title.textContent=`${m.channel_a} → ${m.channel_b}`;
const meter=document.createElement('div');meter.className='cx-match-meter';const bar=document.createElement('i');bar.style.width=String(Math.max(0,m.response_corr)*100)+'%';meter.append(bar);const score=document.createElement('span');score.textContent=f(m.response_corr,2);row.append(title,meter,score);row.addEventListener('click',()=>loadPair(m.channel_a,m.channel_b));$('cx-matches').append(row)}
if(doc.matches.length)loadPair(doc.matches[0].channel_a,doc.matches[0].channel_b);
notify(`Compared ${doc.probe_count} identical held-out probes · bootstrap n=${doc.replicates}.`);
}catch(err){notify('Comparison rejected: '+err.message);text('cx-cka','—');text('cx-ci','—');text('cx-corr','—');$('cx-matches').replaceChildren()}}
function linePath(xs,ys,width,height){const top=16,bottom=height-24,margin=35;return xs.map((x,i)=>`${i?'L':'M'}${(margin+x*(width-margin-16)).toFixed(2)},${(bottom-ys[i]*(bottom-top)).toFixed(2)}`).join(' ')}
function deletionChart(doc){const chart=$('cx-deletion-chart');chart.replaceChildren();const NS='http://www.w3.org/2000/svg';
function svg(name,attrs){const e=document.createElementNS(NS,name);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);chart.append(e);return e}
svg('line',{x1:35,y1:156,x2:424,y2:156,stroke:'#c7cedf'});svg('line',{x1:35,y1:16,x2:35,y2:156,stroke:'#c7cedf'});
const max=Math.max(...doc.top_cam_deletion,...doc.random_deletion_mean,.05);
for(const [vals,color] of [[doc.random_deletion_mean,'#b1b7c6'],[doc.top_cam_deletion,'#7584df']])svg('path',{d:linePath(doc.fractions,vals.map(v=>v/max),440,180),fill:'none',stroke:color,'stroke-width':3,'stroke-linejoin':'round'});
for(const [t,c] of [['CAM order','#7584df'],['Random average','#b1b7c6']]){const e=svg('text',{x:c==='#7584df'?45:205,y:12,fill:c,'font-size':11});e.textContent=t}
const e=svg('text',{x:288,y:175,fill:'#8493ac','font-size':10});e.textContent='fraction occluded →';}
async function loadPair(a,b){const run=state.run,other=state.other,step=state.step,otherStep=state.otherStep,layer=state.layer;
try{const d=await get(query('/api/concepts/matched-examples',{run_a:run,run_b:other,step_a:step,step_b:otherStep,layer,channel_a:a,channel_b:b,k:6}));
if(run!==state.run||other!==state.other||step!==state.step||otherStep!==state.otherStep||layer!==state.layer)return;
text('cx-overlap',`Jaccard ${f(d.jaccard,2)} · ${d.intersection_size}/6 shared`);
for(const [id,examples] of [['cx-pair-a',d.topk_a],['cx-pair-b',d.topk_b]]){const parent=$(id);parent.replaceChildren();for(const sample of examples){const img=document.createElement('img');img.src=sample.input;img.alt=`Sample #${sample.probe_index} · class ${sample.label}`;img.title=img.alt;parent.append(img)}}
}catch(err){notify('Matched sample comparison unavailable: '+err.message)}}
async function test(){const v=state.version;notify('Occluding real image patches and measuring target-class probabilities…');
try{const doc=await get(query('/api/concepts/faithfulness',{run_id:state.run,step:state.step,sample_index:Number($('cx-sample').value),target_class:Number($('cx-target').value)}));
if(v!==state.version)return;deletionChart(doc);text('cx-del-auc',f(doc.auc_top_cam,4));text('cx-random-auc',f(doc.auc_random,4));text('cx-auc-gap',f(doc.auc_random_minus_cam,4));notify(`True class ${doc.label}; predicted class ${doc.model_prediction}. Zero-patch occlusion · ${doc.random_trials} random baselines · target ${doc.target_class}.`);
}catch(err){notify('Occlusion unavailable: '+err.message)}}
$('cx-run').addEventListener('change',selectRun);$('cx-step').addEventListener('change',loadTopk);
$('cx-layer').addEventListener('change',selectLayer);$('cx-channel').addEventListener('change',loadTopk);
$('cx-other').addEventListener('change',()=>selectOther().catch(e=>notify(e.message)));
$('cx-other-step').addEventListener('change',()=>{state.otherStep=Number($('cx-other-step').value)});
$('cx-compare').addEventListener('click',compare);$('cx-test').addEventListener('click',test);
loadRuns();
})();