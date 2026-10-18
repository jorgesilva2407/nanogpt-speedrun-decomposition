"""Extract the training-script source embedded at the top of a modded-nanogpt log.

Two layouts occur in the #11-#18 window (determined by reading the scripts' logging code):
  * r11-r13 style:  f.write('='*100+'\\n'); f.write(code); f.write('='*100+'\\n')
      -> log = SEP + code + SEP ...   (code reproduced byte-exact)
  * r14+ style (print0):  print0(code); print0('='*100)
      -> log = code + '\\n' + SEP ...  (print adds one trailing newline)
Pure Python; no torch.
"""
SEP = '=' * 100


def extract(log_text: str) -> str:
    lines = log_text.split('\n')
    if lines and lines[0] == SEP:
        end = lines.index(SEP, 1)
        body = lines[1:end]
        return '\n'.join(body) + '\n'
    end = lines.index(SEP)
    body = '\n'.join(lines[:end])
    # print0(code) appended one '\n' after code; code itself normally ends in '\n'
    # so body (which excludes the final '\n' before SEP) == code exactly when code ends with '\n'.
    return body


def norm_ws(s: str) -> str:
    """Whitespace-insensitive normal form: strip trailing ws per line, drop blank lines."""
    return '\n'.join(l.rstrip() for l in s.split('\n') if l.strip())
