#!/usr/bin/env python3
"""Offline smoke test of seven main publication figures and ten result tables."""
from pathlib import Path
import tempfile, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as out:
    cmd=[sys.executable,str(ROOT/'publication/rebuild_manuscript_assets.py'),'--output',out]
    r=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True)
    assert r.returncode==0,(r.stdout,r.stderr)
    d=Path(out)
    figures=sorted((d/'figures').glob('Figure_*.png'))
    tables=sorted((d/'tables').glob('*.csv'))
    assert len(figures)==7,figures
    assert len(tables)==10,tables
    assert (d/'MANIFEST.json').is_file()
    assert all(p.stat().st_size>10000 for p in figures)
    print(f'PASS: {len(figures)} figures, {len(tables)} CSV tables, and manifest regenerated offline')
