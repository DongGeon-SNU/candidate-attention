#!/usr/bin/env python3
"""Terminal-token agreement audit using official DAPD and Fast-dLLM functions.

No dependency heuristic is defined here. DEMASK deliberately fails closed until
the authors' predictor code and compatible checkpoint are available.
"""
from __future__ import annotations
import argparse, csv, datetime as dt, json, random, sys, uuid
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]

def dump_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields=sorted({k for r in rows for k in r}) if rows else []
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fields,extrasaction="ignore"); w.writeheader(); w.writerows(rows)
def dump_jsonl(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows),encoding="utf-8")
def js(x): return json.dumps(x,ensure_ascii=False,separators=(",",":"))

def api():
    root=ROOT/"vendor"/"DAPD"
    if not (root/"dapd"/"core.py").is_file(): raise RuntimeError("Official DAPD source missing: run setup on H100.")
    sys.path[:0]=[str(root),str(root/"baselines"/"Fast-dLLM"/"llada")]
    from dapd.core import AttentionCaptureHook, build_dependency_graph, compute_dependency_scores, select_independent_set
    from dapd.generation import (_compute_attention,_compute_confidence_and_predictions,_compute_tau,_resolve_capture_layers,_compute_dapd_direct_selection,_apply_dapd_staged_selection,validate_dapd_algorithm)
    from generate import get_transfer_index
    return locals()
def seed(n):
    random.seed(n); np.random.seed(n); torch.manual_seed(n); torch.cuda.manual_seed_all(n)
    torch.use_deterministic_algorithms(True,warn_only=True); torch.backends.cuda.matmul.allow_tf32=False
def model_and_tokenizer(c):
    from transformers import AutoTokenizer
    from model.modeling_llada import LLaDAModelLM
    m=c["model"]; dtype=torch.bfloat16 if m["dtype"]=="bfloat16" else torch.float16
    t=AutoTokenizer.from_pretrained(m["name"],revision=m["hf_revision"],trust_remote_code=True)
    return LLaDAModelLM.from_pretrained(m["name"],revision=m["hf_revision"],trust_remote_code=True,torch_dtype=dtype).cuda().eval(),t
def ids(tok,p):
    p=tok.apply_chat_template([{"role":"user","content":p}],add_generation_prompt=True,tokenize=False)
    return torch.tensor(tok(p)["input_ids"],device="cuda").unsqueeze(0)

@torch.no_grad()
def fast(model,x,d,transfer):
    x=torch.cat([x,torch.full((1,d["generation_length"]),d["mask_id"],device="cuda",dtype=torch.long)],1); rows=[]
    for step in range(d["steps"]):
        mask=x.eq(d["mask_id"])
        if not mask.any(): break
        logits=model(x,use_cache=False).logits
        proposed,selected=transfer(logits,d["temperature"],d["remasking"],mask,x,None,d["fast_threshold"])
        rows.append({"decode_step":step,"masked_position_count":int(mask.sum()),"selected_positions":selected[0].nonzero().flatten().tolist()})
        x=torch.where(selected,proposed,x)
    if x.eq(d["mask_id"]).any(): raise RuntimeError("Fast-dLLM incomplete; increase steps.")
    return x,rows

def trace(conf,raw,norm,mask,tau):
    scores=raw.sum(-1)*mask.float(); rank=torch.argsort((scores/(scores.max(-1,keepdim=True)[0]+1e-8)*conf/(conf.max(-1,keepdim=True)[0]+1e-8)*mask.float())[0,mask[0]],descending=True)
    positions=mask[0].nonzero().flatten()[rank].tolist(); selected=[]; rejected=[]
    for c in positions:
        blockers=[(s,float(norm[0,c,s])) for s in selected if norm[0,c,s]>tau]
        if blockers:
            rejected += [{"position_i":c,"position_j":s,"attention_dependency_score":score,"edge_threshold":tau,"selection_stage":"greedy_independent_set","exclusion_reason":"normalized_dependency_gt_tau"} for s,score in blockers]
        else: selected.append(c)
    return selected,rejected

@torch.no_grad()
def dapd(model,x,d,c,a):
    alg=a["validate_dapd_algorithm"](c["algorithm"]); x=torch.cat([x,torch.full((1,d["generation_length"]),d["mask_id"],device="cuda",dtype=torch.long)],1)
    h=a["AttentionCaptureHook"](capture_layers=a["_resolve_capture_layers"](model,layer_ratio=c["layer_ratio"])); rows=[]; h.register(model)
    try:
        for step in range(d["steps"]):
            mask=x.eq(d["mask_id"]); n=int(mask.sum())
            if not n: break
            tau=a["_compute_tau"](remaining_masks=n,gen_length=d["generation_length"],tau_min=c["tau_min"],tau_max=c["tau_max"])
            h.clear(); logits=model(x,use_cache=False).logits; attention=a["_compute_attention"](h,c["layer_ratio"])
            raw,norm=a["build_dependency_graph"](attention,mask); ds=a["compute_dependency_scores"](raw,mask); conf,pred,_=a["_compute_confidence_and_predictions"](logits)
            greedy=mask; direct=torch.zeros_like(mask); stage={}
            if alg=="dapd_direct": direct,stage=a["_compute_dapd_direct_selection"](confidence=conf,mask_index=mask); greedy=mask&~direct
            selected=a["select_independent_set"](conf,ds,norm,greedy,tau)|direct
            # Preserve the pre-staged graph decision for integrity checking:
            # staged additions are an official, separate DAPD stage.
            graph_selected=(selected&greedy)[0].nonzero().flatten().tolist()
            if alg=="dapd_staged": selected,stage=a["_apply_dapd_staged_selection"](selected=selected,confidence=conf,mask_index=mask,progress=1-n/d["generation_length"])
            if not selected.any(): selected[0,torch.argmax(torch.where(mask[0],conf[0],torch.tensor(-float("inf"),device="cuda")))]=True; stage["fallback_added"]=1
            replay,rejected=trace(conf,raw,norm,greedy,tau)
            if sorted(replay)!=sorted(graph_selected): raise AssertionError("logging replay changed/reported a different official DAPD selection")
            rows.append({"decode_step":step,"masked_position_count":n,"selected_positions":selected[0].nonzero().flatten().tolist(),"tau":tau,"selection_replay_verified":True,"rejected":rejected,"stage":stage})
            x=torch.where(selected,pred,x)
    finally: h.restore()
    if x.eq(d["mask_id"]).any(): raise RuntimeError("DAPD incomplete; increase steps.")
    return x,rows

def bootstrap(edges,reps,seed_value):
    groups=defaultdict(list)
    for e in edges: groups[e["prompt_id"]].append(e["agreement_count"])
    ps=list(groups); rng=np.random.default_rng(seed_value)
    if not ps:return {str(k):[None,None] for k in range(3)}
    a=[]
    for _ in range(reps):
        v=[x for p in rng.choice(ps,len(ps),replace=True) for x in groups[p]]; a.append([np.mean(np.array(v)==k) for k in range(3)])
    a=np.array(a); return {str(k):np.quantile(a[:,k],[.025,.975]).tolist() for k in range(3)}
def plot(edges,bins,path):
    import matplotlib.pyplot as plt
    scores=np.array([e["attention_dependency_score"] for e in edges]); agree=np.array([e["agreement_count"] for e in edges]); labels=[]; values=[]
    for lo,hi in zip(bins,bins[1:]):
        m=(scores>=lo)&(scores<hi); labels.append(f"[{lo:g},{hi:g})"); values.append(np.mean(agree[m]==2) if m.any() else np.nan)
    fig,ax=plt.subplots(figsize=(8,4)); ax.bar(labels,values);ax.set_ylim(0,1);ax.set_ylabel("2/2 terminal agreement");ax.set_xlabel("DAPD normalized dependency");ax.tick_params(axis="x",rotation=35);fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--allow-missing-demask",action="store_true");z=p.parse_args();c=yaml.safe_load(Path(z.config).read_text(encoding="utf-8"));d=c["decoding"]
    if c.get("demask") and not z.allow_missing_demask: raise RuntimeError("DEMASK requested but no official predictor/checkpoint supplied; refusing a surrogate.")
    if d["temperature"]!=0 or d.get("top_p") is not None: raise RuntimeError("This audit supports only deterministic greedy temperature=0, top_p=null.")
    a=api();seed(d["seed"]);model,tok=model_and_tokenizer(c); prompts=list(c.get("prompts") or [])
    if c.get("prompt_file"):
        q=ROOT/c["prompt_file"]
        if not q.is_file():raise RuntimeError(f"prompt_file unavailable: {q}")
        prompts += [json.loads(x)["prompt"] for x in q.read_text(encoding="utf-8").splitlines() if x]
    if not prompts:raise RuntimeError("No prompts configured.")
    stamp=dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ");rid="sdh-"+uuid.uuid4().hex[:8];out=ROOT/c["storage"]["output_root"]/stamp; raw=out/"results"/"raw"; summary=out/"results"/"summary"; fs=[];ds=[];edges=[];outputs=[]
    for pid,prompt in enumerate(prompts):
        x=ids(tok,prompt); f,fr=fast(model,x,d,a["get_transfer_index"]); q,dr=dapd(model,x,d,c["dapd"],a); fg=f[0,x.shape[1]:].tolist();dg=q[0,x.shape[1]:].tolist();ft=tok.decode(fg,skip_special_tokens=True);dtxt=tok.decode(dg,skip_special_tokens=True)
        outputs += [{"run_id":rid,"prompt_id":pid,"seed":d["seed"],"baseline":"fast_dllm","final_output_token_ids":js(fg),"decoded_text":ft},{"run_id":rid,"prompt_id":pid,"seed":d["seed"],"baseline":"dapd","final_output_token_ids":js(dg),"decoded_text":dtxt,"fast_output_token_ids":js(fg),"fast_decoded_text":ft}]
        fs += [{"run_id":rid,"prompt_id":pid,"seed":d["seed"],"baseline":"fast_dllm",**{k:js(v) if isinstance(v,list) else v for k,v in r.items()},"final_output_token_ids":js(fg),"decoded_text":ft} for r in fr]
        for r in dr:
            ds.append({"run_id":rid,"prompt_id":pid,"seed":d["seed"],"baseline":"dapd","decode_step":r["decode_step"],"masked_position_count":r["masked_position_count"],"selected_positions":js(r["selected_positions"]),"tau":r["tau"],"selection_replay_verified":r["selection_replay_verified"],"final_output_token_ids":js(dg),"decoded_text":dtxt,"fast_output_token_ids":js(fg),"fast_decoded_text":ft})
            for e in r["rejected"]:
                i,j=e["position_i"]-x.shape[1],e["position_j"]-x.shape[1]
                if i>=0 and j>=0:
                    mi,mj=dg[i]==fg[i],dg[j]==fg[j];edges.append({"run_id":rid,"prompt_id":pid,"seed":d["seed"],"baseline":"dapd","decode_step":r["decode_step"],**e,"position_i_generation":i,"position_j_generation":j,"dapd_token_i":dg[i],"dapd_token_j":dg[j],"fast_token_i":fg[i],"fast_token_j":fg[j],"match_i":mi,"match_j":mj,"match_both":mi and mj,"agreement_count":int(mi)+int(mj)})
    for name,rows in [("fast_dllm_steps",fs),("dapd_steps",ds),("dapd_excluded_edges",edges),("final_outputs",outputs)]:dump_jsonl(raw/(name+".jsonl"),rows);dump_csv(raw/(name+".csv"),rows)
    n=len(edges);count=Counter(e["agreement_count"] for e in edges); row={"baseline":"dapd","rejection_events":n,"unique_prompts":len({e["prompt_id"] for e in edges}),"unique_position_pairs":len({(e["prompt_id"],e["position_i_generation"],e["position_j_generation"]) for e in edges}),**{f"agreement_{k}_rate":count[k]/n if n else None for k in range(3)},"prompt_bootstrap_95_ci":bootstrap(edges,c["analysis"]["bootstrap_replicates"],c["analysis"]["bootstrap_seed"])}
    dump_csv(summary/"baseline_summary.csv",[row]);summary.mkdir(parents=True,exist_ok=True);(summary/"baseline_summary.json").write_text(json.dumps(row,indent=2),encoding="utf-8")
    if edges:plot(edges,c["analysis"]["dependency_bins"],summary/"dapd_dependency_agreement.png")
    (summary/"run_manifest.json").write_text(json.dumps({"run_id":rid,"model":c["model"],"decoding":d,"dapd":c["dapd"],"demask_status":"blocked_no_official_predictor_or_checkpoint","selection_integrity":"official functions plus asserted post-hoc replay"},indent=2),encoding="utf-8")
    print(json.dumps({"run_id":rid,"output":str(out),"dapd_rejection_events":n},indent=2))
if __name__=="__main__":main()
