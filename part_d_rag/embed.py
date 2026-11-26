import torch
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModel

from paths import INDEX_PATH,load_data
from chunking import build_chunks


def load_embedder(device):
    name="sentence-transformers/all-MiniLM-L6-v2"
    tokenizer=AutoTokenizer.from_pretrained(name)
    model=AutoModel.from_pretrained(name).to(device)
    model.eval()
    return tokenizer,model


def embed(texts,tokenizer,model,batch_size=256):
    device=next(model.parameters()).device
    out=[]

    with torch.no_grad():
        for start in range(0,len(texts),batch_size):
            batch=texts[start:start+batch_size]
            tok=tokenizer(batch,padding=True,truncation=True,max_length=256,return_tensors="pt").to(device)

            h=model(**tok).last_hidden_state                 #[B, T, 384]

            #mean pooling with the mask, padding tokens get multiplied by 0
            m=tok["attention_mask"].unsqueeze(-1).float()    #[B, T, 1]
            emb=(h*m).sum(1)/m.sum(1)

            emb=F.normalize(emb,dim=-1)
            out.append(emb.cpu())

    return torch.cat(out,dim=0)


def main():
    device="cuda" if torch.cuda.is_available() else "cpu"

    corpus,_=load_data()
    chunk_texts,chunk_abs_ids=build_chunks(corpus)
    print(f"Embedding {len(chunk_texts)} chunks on {device}")

    tokenizer,model=load_embedder(device)
    E=embed(chunk_texts,tokenizer,model)
    print("Index shape:",tuple(E.shape))

    #check: every vector should have length 1
    norms=E.norm(dim=1)
    assert torch.allclose(norms,torch.ones_like(norms),atol=1e-4),"embeddings are not normalized"
    print(f"CHECK embedding norms: min {norms.min():.6f}, max {norms.max():.6f} (all = 1)")

    torch.save({"emb":E,"abs_ids":chunk_abs_ids,"texts":chunk_texts},INDEX_PATH)
    print("Saved index to",INDEX_PATH)


if __name__=="__main__":
    main()