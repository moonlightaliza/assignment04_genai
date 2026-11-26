import os
import json
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM

from paths import RES,INDEX_PATH,load_data

SYSTEM=("You are a biomedical question answering assistant. "
        "Answer the question using only the numbered passages provided. "
        "Cite the passages you use like [2]. "
        "If the passages do not contain the answer, say \"I don't know\".")


def load_generator(device):
    name="Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer=AutoTokenizer.from_pretrained(name)
    model=AutoModelForCausalLM.from_pretrained(name,torch_dtype=torch.float16).to(device)
    model.eval()
    return tokenizer,model


def build_messages(question,passages):
    ctx="\n".join(f"[{i+1}] {p}" for i,p in enumerate(passages))
    user=f"Passages:\n{ctx}\n\nQuestion: {question}"
    return [{"role":"system","content":SYSTEM},{"role":"user","content":user}]


def answer(question,passages,tokenizer,model):
    device=next(model.parameters()).device
    messages=build_messages(question,passages)
    #chat template as text first, then tokenize: works the same across transformers versions
    prompt=tokenizer.apply_chat_template(messages,add_generation_prompt=True,tokenize=False)
    enc=tokenizer(prompt,return_tensors="pt").to(device)

    with torch.no_grad():
        out=model.generate(**enc,do_sample=False,max_new_tokens=256)

    #decode only the new tokens, otherwise the answer includes the whole prompt
    new=out[0][enc["input_ids"].shape[1]:]
    return tokenizer.decode(new,skip_special_tokens=True).strip()


def main():
    device="cuda"
    _,questions=load_data()
    questions=questions[:100]

    idx=torch.load(INDEX_PATH)
    chunk_texts=idx["texts"]
    chunk_abs_ids=idx["abs_ids"]

    with open(os.path.join(RES,"retrieval.json")) as f:
        ret=json.load(f)
    by_qid={r["qid"]:r for r in ret}

    tokenizer,model=load_generator(device)

    with open(os.path.join(RES,"rag_answers.jsonl"),"w") as f:
        for i,q in enumerate(questions):
            top=by_qid[q["qid"]]["top_chunks"]       #5 chunk indices
            passages=[chunk_texts[c] for c in top]
            ans=answer(q["question"],passages,tokenizer,model)

            row={"qid":q["qid"],
                 "question":q["question"],
                 "answer":ans,
                 "reference":q["answer"],
                 "passages":passages,
                 "chunk_abs_ids":[chunk_abs_ids[c] for c in top],
                 "relevant_ids":list(q["relevant_ids"])}
            f.write(json.dumps(row)+"\n")
            if (i+1)%10==0:
                print("Answered",i+1)


if __name__=="__main__":
    main()