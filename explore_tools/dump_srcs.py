"""Write each record's embedded log source to <outdir>/rNN.log.py (byte-exact as extracted)."""
import os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logsrc import extract
repo, out = sys.argv[1:3]
LOGS = {11: 'records/111024_UNetDoubleLr/c87bb826-797b-4f37-98c7-d3a5dad2de74.txt',
        12: 'records/111924_FlexAttention/8384493d-dba9-4991-b16b-8696953f5e6d.txt',
        13: 'records/112424_WindowWarmup/cf9e4571-c5fc-4323-abf3-a98d862ec6c8.txt',
        14: 'records/120424_ValueEmbed/00008ea0-21dd-442a-82ee-d12799249d0f.txt',
        15: 'records/120824_UNetValueEmbedsTweaks/0069607b-aa90-49fd-9766-4368bcd168c4.txt',
        16: 'records/121024_MFUTweaks/04df8c70-2462-4b6c-9e49-c7589b321a81.txt',
        17: 'records/121724_SparsifyEmbeds/165384d5-5a41-4fba-9b79-8ecc372616ea.txt',
        18: 'records/010425_SoftCap/31d6c427-f1f7-4d8a-91be-a67b5dcd13fd.txt'}
os.makedirs(out, exist_ok=True)
for n, p in LOGS.items():
    txt = subprocess.run(['git', '-C', repo, 'show', f'5deb47d:{p}'], capture_output=True, text=True, check=True).stdout
    open(os.path.join(out, f'r{n}.log.py'), 'w').write(extract(txt))
    print(n, extract(txt).count('\n'))
