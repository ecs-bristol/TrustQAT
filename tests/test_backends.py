import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from trustqat.backends import (
    API_VERSION,
    BaseBackend,
    CapabilityError,
    api_compatible,
    available_backends,
    get_backend,
    register_backend,
    require_capability,
)

SRC = Path(__file__).resolve().parents[1] / 'src' / 'trustqat'


def test_api_compatible():
    assert api_compatible(API_VERSION)
    assert not api_compatible('0.0')
    assert not api_compatible('1.0')


def test_hgq2_registered_through_entry_point():
    assert available_backends().get('hgq2') == 'trustqat.backends.hgq2:HGQ2Backend'


def test_runtime_registration_and_capabilities():
    class Dummy(BaseBackend):
        name = 'dummy'
        framework = 'keras'
        cost_names = ('x',)

        def cost(self, model):
            return {'x': 0.0}

        def save(self, model, path):
            pass

        def load(self, path, cfg=None):
            return None

    register_backend('dummy', Dummy)
    b = get_backend('dummy')
    assert 'dummy' in available_backends()
    with pytest.raises(CapabilityError):
        require_capability(b, 'freeze_precision')
    with pytest.raises(CapabilityError):
        b.freeze_precision(None)


def test_incompatible_api_version_rejected():
    class Old(BaseBackend):
        name = 'old'
        framework = 'keras'
        api_version = '0.0'

    register_backend('old', Old)
    with pytest.raises(RuntimeError, match='API'):
        get_backend('old')


def test_core_does_not_import_frameworks():
    code = (
        'import sys, trustqat, trustqat.core, trustqat.objectives, trustqat.backends, trustqat.models; '
        "bad = [m for m in ('keras', 'torch', 'hgq', 'jax', 'tensorflow') if m in sys.modules]; "
        'assert not bad, bad'
    )
    subprocess.run([sys.executable, '-c', code], check=True)


def test_hgq_only_in_its_backend():
    offenders = []
    for path in SRC.rglob('*.py'):
        if path.name == 'hgq2.py' and path.parent.name == 'backends':
            continue
        text = path.read_text()
        if 'import hgq' in text or 'from hgq' in text:
            offenders.append(str(path.relative_to(SRC)))
    assert not offenders, f'HGQ imported outside backends/hgq2.py: {offenders}'


@pytest.mark.skipif(importlib.util.find_spec('hgq') is None, reason='HGQ2 not installed')
def test_hgq2_backend_loads():
    b = get_backend('hgq2')
    assert b.framework == 'keras' and 'trainable_precision' in b.capabilities and b.cost_names == ('ebops',)
