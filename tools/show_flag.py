"""Print every hunk, across all records, that maps to a switch (flag name or config:<key>).
Usage: python -I tools/show_flag.py <flag> [--no-body]"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from changes_report import load, as_list

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('flag'); ap.add_argument('--no-body', action='store_true')
    a = ap.parse_args()
    n = 0
    for r in load():
        for e in r['hunks']:
            flags = [str(f) for f in as_list(e['flag'])]
            parts = [p for p in e.get('parts', []) if str(p.get('flag')) == a.flag]
            if a.flag not in flags and not parts:
                continue
            n += 1
            print(f"=== r{r['record']}/{e['id']}  {e['file']} {e['header']}  [{', '.join(as_list(e['category']))}] "
                  f"provenance={e['provenance']} numerics={e['numerics']} bundled={e['bundled']}")
            print(f"    {e['description']}")
            for p in parts:
                print(f"    part: {p.get('what')}")
            if e.get('note'):
                print(f"    note: {e['note']}")
            if not a.no_body:
                for l in e['_body']:
                    if l[:1] in '+-':
                        print('    ' + l)
    print(f'{n} hunks map to {a.flag}')
    sys.exit(0 if n else 1)

if __name__ == '__main__':
    main()
