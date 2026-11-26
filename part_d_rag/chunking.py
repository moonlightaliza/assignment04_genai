from paths import load_data


def chunk_abstract(text,size=150,overlap=30):
    words=text.split()
    step=size-overlap    #120
    chunks=[]
    start=0

    while True:
        chunks.append(" ".join(words[start:start+size]))
        #stop once a chunk reaches the end, otherwise the last chunk would be tiny and fully inside the previous one
        if start+size>=len(words):
            break
        start+=step

    return chunks


def build_chunks(corpus):
    """corpus is {abstract_id: text} from bioasq.load_corpus()."""
    chunk_texts=[]
    chunk_abs_ids=[]

    for abs_id,text in corpus.items():
        for c in chunk_abstract(text):
            chunk_texts.append(c)
            chunk_abs_ids.append(abs_id)   #every chunk remembers its abstract

    return chunk_texts,chunk_abs_ids


def main():
    corpus,questions=load_data()
    chunk_texts,chunk_abs_ids=build_chunks(corpus)

    #Table 1
    print("Abstracts:",len(corpus))
    print("Chunks:",len(chunk_texts))
    mean_rel=sum(len(q["relevant_ids"]) for q in questions)/len(questions)
    print("Test questions:",len(questions))
    print("Mean relevant abstracts per question:",round(mean_rel,2))

    #check: no chunk longer than 150 words
    longest=max(len(c.split()) for c in chunk_texts)
    assert longest<=150,"a chunk is longer than 150 words"
    assert len(chunk_abstract("word "*150))==1,"<=150 words must give one chunk"
    #check: consecutive chunks share exactly 30 words
    two=chunk_abstract(" ".join(str(i) for i in range(200)))
    assert two[0].split()[-30:]==two[1].split()[:30],"overlap is not 30 words"
    print(f"CHECK longest chunk: {longest} words (<= 150); overlap = 30 words")


if __name__=="__main__":
    main()