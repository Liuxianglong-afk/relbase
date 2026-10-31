import sys, re, urllib.parse, urllib.request, xml.etree.ElementTree as ET, json
from collections import Counter, defaultdict
B="https://fcp-indi.s3.us-east-1.amazonaws.com"; NS={"s":"http://s3.amazonaws.com/doc/2006-03-01/"}
def keys(prefix):
    tok=None
    while True:
        q={"list-type":"2","prefix":prefix,"max-keys":"1000"}
        if tok: q["continuation-token"]=tok
        r=ET.fromstring(urllib.request.urlopen(f"{B}/?{urllib.parse.urlencode(q)}",timeout=120).read())
        for c in r.findall("s:Contents",NS): yield c.find("s:Key",NS).text,int(c.find("s:Size",NS).text)
        if r.findtext("s:IsTruncated",namespaces=NS)!="true": return
        tok=r.findtext("s:NextContinuationToken",namespaces=NS)
out={}
# DCAN ABIDE I: subjects with parcellated timeseries, per atlas
for coh,pre in [("abide1","data/Projects/ABIDE/Derivatives/DCAN/"),("abide2","data/Projects/ABIDE2/Derivatives/DCAN/")]:
    atl=defaultdict(set); mot=set(); gb=0; allsub=set(); ex=[]
    for k,s in keys(pre):
        rel=k[len(pre):]; m=re.match(r"(sub-[^/]+)/",rel)
        if not m: continue
        allsub.add(m[1])
        a=re.search(r"atlas-([A-Za-z0-9]+)_.*ptseries\.nii$",rel)
        if a: atl[a[1]].add(m[1]); gb+=s
        if rel.endswith("motion_mask.mat"): mot.add(m[1])
        if len(ex)<6 and "ptseries" in rel: ex.append(rel)
    out[f"dcan_{coh}"]={"n_subject_dirs":len(allsub),"ptseries_by_atlas":{a:len(v) for a,v in sorted(atl.items())},"n_motion_mask":len(mot),"ptseries_GB_all_atlases":round(gb/1e9,2),"examples":ex}
    print(coh,json.dumps(out[f"dcan_{coh}"])[:900],flush=True)
# PCP ABIDE I: pipelines x strategies x rois_cc200 counts
pcp={}
for pipe in ["cpac","dparsf","niak","ccs"]:
    for st in ["filt_global","filt_noglobal","nofilt_global","nofilt_noglobal"]:
        n=sum(1 for k,s in keys(f"data/Projects/ABIDE_Initiative/Outputs/{pipe}/{st}/rois_cc200/") if k.endswith(".1D"))
        pcp[f"{pipe}/{st}"]=n
print("pcp_cc200",pcp,flush=True)
out["pcp_abide1_rois_cc200_counts"]=pcp
json.dump(out,open(sys.argv[1],"w",encoding="utf-8"),ensure_ascii=False,indent=1)
