import os
import re
import csv
import json
import random
from rouge_score import rouge_scorer

from paths import RES

IDK=["don't know","do not know","does not contain","do not contain","not provided"]
MEMBERS=["Aliza",]


def cited_numbers(text):
    #matches [2], [1, 3] and [1][3] (the last one is just two matches)
    nums=[]
    for m in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]",text):
        nums+=[int(x) for x in m.split(",")]
    return nums


def is_idk(text):
    t=text.lower().replace("’","'")   #curly apostrophes normalised first
    return any(p in t for p in IDK)


def main():
    with open(os.path.join(RES,"rag_answers.jsonl")) as f:
        rows=[json.loads(line) for line in f]

    #ROUGE-L
    scorer=rouge_scorer.RougeScorer(["rougeL"])
    rouge=sum(scorer.score(r["reference"],r["answer"])["rougeL"].fmeasure for r in rows)/len(rows)

    #citation precision, correct citations over all citations across all answers
    correct=0
    total=0
    for r in rows:
        R=set(r["relevant_ids"])
        for n in cited_numbers(r["answer"]):
            total+=1
            if 1<=n<=len(r["chunk_abs_ids"]) and r["chunk_abs_ids"][n-1] in R:
                correct+=1
    cit_prec=correct/total if total else 0.0

    #"I don't know" rate
    idk=sum(is_idk(r["answer"]) for r in rows)/len(rows)

    print(f"ROUGE-L: {rouge:.4f}")
    print(f"Citation precision: {cit_prec:.4f} ({correct}/{total})")
    print(f"I don't know rate: {idk:.4f}")

    with open(os.path.join(RES,"rag_generation_metrics.json"),"w") as f:
        json.dump({"rougeL":rouge,"citation_precision":cit_prec,"citations_correct":correct,
                   "citations_total":total,"idk_rate":idk},f,indent=2)

    #human-eval sheet, 20 random answers (fixed seed). Kept separate from the diffusion
    #blind test; merge both into results/human_eval.csv at the end.
    pick=random.Random(42).sample(range(len(rows)),20)
    header=["qid","question","answer","reference","passages"]
    for m in MEMBERS:
        header+=[f"supported_{m}",f"correct_{m}"]
    with open(os.path.join(RES,"human_eval_rag.csv"),"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(header)
        for i in pick:
            r=rows[i]
            passages="\n\n".join(f"[{j+1}] {p}" for j,p in enumerate(r.get("passages",[])))
            w.writerow([r["qid"],r["question"],r["answer"],r["reference"],passages]+[""]*(2*len(MEMBERS)))
    print("Wrote human_eval_rag.csv")


if __name__=="__main__":
    main()