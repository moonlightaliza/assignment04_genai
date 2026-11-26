import os
import json
import torch

from paths import RES,INDEX_PATH,load_data
from embed import load_embedder,embed


def load_index(device):
    idx=torch.load(INDEX_PATH)
    E=idx["emb"].to(device)                   #[N_chunks, 384]
    return E,idx["abs_ids"],idx["texts"]


def search(q_emb,E,chunk_abs_ids,k_abs=20,k_chunk=5,batch_size=64):
    device=E.device

    #map each abstract id to a position 0...n_abstracts-1
    abs_list=list(dict.fromkeys(chunk_abs_ids))
    pos={a:i for i,a in enumerate(abs_list)}
    chunk_pos=torch.tensor([pos[a] for a in chunk_abs_ids],dtype=torch.long,device=device)   #[N_chunks]

    top_abs=[]
    top_chunks=[]

    for start in range(0,q_emb.shape[0],batch_size):
        q=q_emb[start:start+batch_size].to(device)       #[Q, 384]
        Q=q.shape[0]

        scores=q@E.T                                     #[Q, N_chunks], cosine similarity

        #an abstract's score is the max over its chunks
        abs_scores=torch.full((Q,len(abs_list)),float("-inf"),device=device)
        abs_scores.scatter_reduce_(1,chunk_pos.expand(Q,-1),scores,reduce="amax")

        idx=abs_scores.topk(k_abs,dim=1).indices.cpu().tolist()
        top_abs+=[[abs_list[p] for p in row] for row in idx]

        top_chunks+=scores.topk(k_chunk,dim=1).indices.cpu().tolist()

    return top_abs,top_chunks


def main():
    device="cuda" if torch.cuda.is_available() else "cpu"

    _,questions=load_data()
    E,chunk_abs_ids,chunk_texts=load_index(device)

    tokenizer,model=load_embedder(device)
    q_emb=embed([q["question"] for q in questions],tokenizer,model)   #questions encoded the same way as chunks

    top_abs,top_chunks=search(q_emb,E,chunk_abs_ids)

    out=[]
    for q,a,c in zip(questions,top_abs,top_chunks):
        out.append({"qid":q["qid"],"ranked":a,"top_chunks":c})

    with open(os.path.join(RES,"retrieval.json"),"w") as f:
        json.dump(out,f)
    print("Searched",len(out),"questions")


if __name__=="__main__":
    main()