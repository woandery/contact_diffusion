"""Process-local validator adaptation: execute every external particle in ID order."""
import inspect
import textwrap
import runpy
from pathlib import Path

import isaacgym  # Import before torch/task modules.
from isaacgymenvs.tasks.fetch.fetch_ptd_dro_render import FetchPtdDRORender

original=FetchPtdDRORender.validate_lifts
source=textwrap.dedent(inspect.getsource(original))
anchor='    ranked = ranked[:count]\n'
assert source.count(anchor)==1
source=source.replace(anchor,
    '    assert external_mode, "Paired study accepts only saved external particles"\n'
    '    assert len(scored) == count, "All particles must be executed, without Top-K selection"\n'
    '    ranked = sorted(scored, key=lambda item: int(item["index"]))\n'
    '    assert [int(item["index"]) for item in ranked] == list(range(count))\n')
namespace=dict(original.__globals__)
exec(compile(source,'<fixed-particle-order-validation>','exec'),namespace)
FetchPtdDRORender.validate_lifts=namespace['validate_lifts']

if __name__=='__main__':
    # Preserve the original __main__/__file__ context used by Hydra to resolve
    # its relative config directory; importing the decorated entry loses it.
    entry=Path(inspect.getfile(original)).parents[2]/'validate_dro_lift.py'
    runpy.run_path(str(entry),run_name='__main__')
