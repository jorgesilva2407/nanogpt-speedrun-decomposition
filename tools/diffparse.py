"""Parse a unified diff (as written by tools/make_diff.py) into hunks. Pure Python."""
import re

HDR = re.compile(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@')

def hunks(text):
    """Yield dicts: file, header, old (start, len), new (start, len), body (list of lines)."""
    out, cur_file, cur, old_file = [], None, None, None
    for line in text.split('\n'):
        if line.startswith('--- '):
            cur = None
            old_file = line[4:].strip()
            old_file = old_file[2:] if old_file.startswith('a/') else old_file
            continue
        if line.startswith('+++ '):
            cur_file = line[4:].strip()
            cur_file = cur_file[2:] if cur_file.startswith('b/') else cur_file
            if cur_file == '/dev/null':      # deleted file: name it by its old path
                cur_file = old_file
            continue
        if line.startswith('Binary files '):
            m = re.match(r'Binary files a/(.*) and b/.* differ', line)
            out.append(dict(file=m.group(1), header='Binary files differ', old=None, new=None, body=[line]))
            cur = None
            continue
        m = HDR.match(line)
        if m:
            o1, o2, n1, n2 = m.groups()
            cur = dict(file=cur_file, header=line.split(' @@')[0] + ' @@',
                       old=[int(o1), int(o2) if o2 is not None else 1],
                       new=[int(n1), int(n2) if n2 is not None else 1], body=[])
            out.append(cur)
            continue
        if cur is not None and line[:1] in (' ', '+', '-', '\\'):
            cur['body'].append(line)
    return out
