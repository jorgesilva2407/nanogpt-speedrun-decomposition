"""The committed table is exactly what the committed saved_results/ produce (the deliverable's reproduction check)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tables

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_reports_table_is_rebuilt_from_saved_results(tmp_path):
    text = tables.build(os.path.join(ROOT, 'saved_results', 'h100'), str(tmp_path / 'table.md'),
                        os.path.join(ROOT, 'saved_results', 'a100'))
    assert text == open(os.path.join(ROOT, 'reports', 'table.md')).read()
