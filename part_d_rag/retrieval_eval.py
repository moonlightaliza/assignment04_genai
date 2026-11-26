import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paths import RES,FIG,load_data


def hit_at_k(ranked,R,k):
    return 1.0 if any(a in R for a in ranked[:k]) else 0.0


def recall_at_k(ranked,R,k):
    return len(set(ranked[:k])&R)/min(k,len(R))


def mrr_at_10(ranked,R):
    for i,a in enumerate(ranked[:10]):
        if a in R:
            return 1.0/(i+1)
    return 0.0


def evaluate(rankings,rel_sets):
    n=len(rankings)
    res={}
    for k in (1,5,10):
        res[f"Hit@{k}"]=sum(hit_at_k(r,R,k) for r,R in zip(rankings,rel_sets))/n
        res[f"Recall@{k}"]=sum(recall_at_k(r,R,k) for r,R in zip(rankings,rel_sets))/n
    res["MRR@10"]=sum(mrr_at_10(r,R) for r,R in zip(rankings,rel_sets))/n
    return res


def main():
    _,questions=load_data()
    with open(os.path.join(RES,"retrieval.json")) as f:
        ret=json.load(f)
    by_qid={r["qid"]:r for r in ret}

    rankings=[by_qid[q["qid"]]["ranked"] for q in questions]
    rel_sets=[set(q["relevant_ids"]) for q in questions]

    #check: a perfect ranking (relevant ids first) must give exactly 1 everywhere
    perfect=[list(R)+[-(i+1) for i in range(20)] for R in rel_sets]
    p=evaluate(perfect,rel_sets)
    for k in (1,5,10):
        assert p[f"Hit@{k}"]==1 and abs(p[f"Recall@{k}"]-1)<1e-9,"perfect ranking does not give 1, check the Recall denominator"
    assert abs(p["MRR@10"]-1)<1e-9,"perfect ranking does not give MRR 1"
    print("CHECK perfect ranking:",{k:round(v,4) for k,v in p.items()})

    #Table 4
    res=evaluate(rankings,rel_sets)
    for name,v in res.items():
        print(f"{name}: {v:.4f}")
    with open(os.path.join(RES,"retrieval_metrics.json"),"w") as f:
        json.dump(res,f,indent=2)

    #Recall@k plot, k=1...20
    ks=list(range(1,21))
    rec=[sum(recall_at_k(r,R,k) for r,R in zip(rankings,rel_sets))/len(rankings) for k in ks]
    plt.figure(figsize=(6,4))
    plt.plot(ks,rec,marker="o")
    plt.xlabel("k")
    plt.ylabel("Recall@k")
    plt.title("RAG retrieval: Recall@k on BioASQ test")
    plt.xticks(ks)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG,"rag_recall_at_k.png"),dpi=200)
    plt.close()

    #failure cases for E.6: Hit@10 = 0
    fails=[]
    for q,r,R in zip(questions,rankings,rel_sets):
        if hit_at_k(r,R,10)==0:
            fails.append({"qid":q["qid"],"question":q["question"],"relevant_ids":sorted(R),"top10":r[:10]})
    with open(os.path.join(RES,"rag_failure_cases.json"),"w") as f:
        json.dump(fails,f,indent=2)
    print("Failure cases (Hit@10=0):",len(fails),"of",len(questions))


if __name__=="__main__":
    main()