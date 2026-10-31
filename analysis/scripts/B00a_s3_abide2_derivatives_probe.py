import sys, re, urllib.parse, urllib.request, xml.etree.ElementTree as ET, json
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
out={}
for name,pre,pat in [("abide2_fmriprep_boldpreproc","data/Projects/ABIDE2/Outputs/fmriprep/fmriprep/",r"sub-(\d+)/.*func/.*desc-preproc_bold\.nii\.gz$|sub-(\d+)/.*func/.*preproc.*bold.*\.nii\.gz$"),
                     ("abide2_denoise_bold","data/Projects/ABIDE2/Outputs/denoise/denoise/",r"sub-(\d+)/.*func/.*bold_denoised\.nii\.gz$"),
                     ("abide2_dcan_any","data/Projects/ABIDE2/Derivatives/DCAN/",r"sub-(\d+)/")]:
    subs=set(); ext=Counter(); tot=0; ex=[]
    for k,s in keys(pre):
        m=re.search(pat,k[len(pre):])
        if m:
            subs.add(next(g for g in m.groups() if g)); tot+=s
        ext[re.sub(r".*?(\.[A-Za-z0-9]+(\.gz)?)$",r"\1",k)]+=1
        if name=="abide2_dcan_any" and len(ex)<25 and "sub-<ID>" in k: ex.append(f"{k[len(pre):]} {s}")
    out[name]={"n_subjects":len(subs),"GB":round(tot/1e9,1),"ext_top":ext.most_common(8)}
    if ex: out[name]["example_files"]=ex
    print(name,json.dumps(out[name],ensure_ascii=False),flush=True)
json.dump(out,open(sys.argv[1],"w",encoding="utf-8"),ensure_ascii=False,indent=1)
