#!/usr/bin/env python3
"""Verify SHA-256 identities of the public research release files."""
from pathlib import Path
import hashlib,argparse,json

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    args=p.parse_args();root=args.root.resolve()
    expected=json.loads((root/'FILE_IDENTITIES.json').read_text())['files']
    failures=[]
    for rel,record in expected.items():
        f=root/rel
        if not f.is_file(): failures.append([rel,'MISSING'])
        elif f.stat().st_size!=record['size_bytes'] or sha256(f)!=record['sha256']:
            failures.append([rel,'HASH_MISMATCH'])
    result={'status':'PASS' if not failures else 'FAIL','files':len(expected),'failures':failures}
    print(json.dumps(result,indent=2))
    return 0 if not failures else 2
if __name__=='__main__':raise SystemExit(main())
