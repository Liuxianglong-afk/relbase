import sys, re, urllib.parse, urllib.request, xml.etree.ElementTree as ET, json
import pandas as pd
from collections import Counter
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
qc=pd.read_csv(sys.argv[1]); ok=set(zip(qc.loc[qc.pass_fd20,"cohort"],qc.loc[qc.pass_fd20,"sub_n"]))
allq=set(zip(qc.cohort,qc.sub_n))
out={}
for coh,pre in [("abide1","data/Projects/ABIDE/Derivatives/DCAN/"),("abide2","data/Projects/ABIDE2/Derivatives/DCAN/")]:
    runs={}; size=[]
    for k,s in keys(pre):
        m=re.match(r"sub-(\d+)/[^/]+/func/.*_run-(\d+)_bold_timeseries\.dtseries\.nii$",k[len(pre):])
        if m: runs.setdefault(int(m[1]),set()).add(int(m[2])); size.append(s)
    nr=Counter(len(v) for v in runs.values())
    two_all=[s for s,v in runs.items() if len(v)>=2 and (coh,s) in allq]
    two_ok=[s for s in two_all if (coh,s) in ok]
    out[coh]={"subjects_by_n_runs":dict(nr),"n_eligible_with_2plus_runs":len(two_all),"n_qcpass_with_2plus_runs":len(two_ok),"median_run_dtseries_MB":round(sorted(size)[len(size)//2]/1e6,1) if size else None}
    sites=Counter(qc.loc[(qc.cohort==coh)&qc.sub_n.isin(two_all),"site"]); out[coh]["sites_2plus_runs_eligible"]=dict(sites)
    print(coh,json.dumps(out[coh],ensure_ascii=False),flush=True)
json.dump(out,open(sys.argv[2],"w",encoding="utf-8"),ensure_ascii=False,indent=1)
