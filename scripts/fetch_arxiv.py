"""Download arXiv abstracts into corpus.jsonl (run this on YOUR machine; needs internet).

  python scripts/fetch_arxiv.py --query "cat:cs.IR" --n 2000 --out data/arxiv/corpus.jsonl
Tip: pick a narrow slice (cs.IR + cs.CL, or one subtopic) so your queries have real near-duplicate
distractors. A corpus of 2k-20k docs is enough to show relevance problems and still run fast.
"""
import argparse
import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

NS = {"a": "http://www.w3.org/2005/Atom"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--query", default="cat:cs.IR")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--out", default="data/arxiv/corpus.jsonl")
    args = p.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    seen, start = set(), 0
    with open(args.out, "w", encoding="utf-8") as f:
        while len(seen) < args.n:
            url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(
                {"search_query": args.query, "start": start, "max_results": 100, "sortBy": "submittedDate"}
            )
            root = ET.fromstring(urllib.request.urlopen(url, timeout=60).read())
            entries = root.findall("a:entry", NS)
            if not entries:
                break
            for e in entries:
                aid = e.find("a:id", NS).text.rsplit("/", 1)[-1]
                if aid in seen:
                    continue
                seen.add(aid)
                title = " ".join(e.find("a:title", NS).text.split())
                text = " ".join(e.find("a:summary", NS).text.split())
                f.write(json.dumps({"id": aid, "title": title, "text": text}) + "\n")
            start += 100
            print(f"{len(seen)} docs", flush=True)
            time.sleep(3)  # arXiv API etiquette: one request per 3 seconds


if __name__ == "__main__":
    main()
