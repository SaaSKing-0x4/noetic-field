"""
Build hook: makes `pip install -e .` expose this folder as the `noetic_field` package.

Why it exists. The repository root is the package folder itself (the modules sit next to pyproject.toml), so the wheel
maps the files into a noetic_field/ folder. hatchling does that for a normal build, but for an editable install it
cannot: it exposes every top-level file as a top-level module (agents, state, api, ...) and refuses a path rewrite that
adds a prefix. Without help, `pip install -e .` would leave `import noetic_field` failing.

What it does. For an editable build only, it adds two small files to the editable wheel: a .pth file and a module that
registers an import finder mapping the name `noetic_field` to this folder's __init__.py, with this folder as its search
path. The files are not copied: `import noetic_field` and `import noetic_field.state` read the files in this folder,
so an edit shows at once. The finder uses the standard library only, and nothing else in the folder is put on
sys.path, so names such as `state` or `api` do not become importable in the environment.

A normal build (pip install ., python -m build) is not touched by this file.
"""

import os
import shutil
import tempfile

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

PACKAGE = "noetic_field"

FINDER = '''\
import os
import sys
from importlib.util import spec_from_file_location

_ROOT = {root!r}
_NAME = {name!r}


class _Finder:
    @classmethod
    def find_spec(cls, fullname, path=None, target=None):
        if fullname != _NAME or path is not None:
            return None
        return spec_from_file_location(
            fullname, os.path.join(_ROOT, "__init__.py"), submodule_search_locations=[_ROOT]
        )

    @classmethod
    def invalidate_caches(cls):
        pass


if _Finder not in sys.meta_path:
    sys.meta_path.append(_Finder)
'''


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        if version != "editable":
            return
        self._tmp = tempfile.mkdtemp(prefix="noetic-field-editable-")
        module = f"_{PACKAGE}_editable"
        py_path = os.path.join(self._tmp, module + ".py")
        pth_path = os.path.join(self._tmp, module + ".pth")
        with open(py_path, "w", encoding="utf-8") as f:
            f.write(FINDER.format(root=os.path.abspath(self.root), name=PACKAGE))
        with open(pth_path, "w", encoding="utf-8") as f:
            f.write(f"import {module}\n")
        build_data["force_include_editable"][py_path] = f"{module}.py"
        build_data["force_include_editable"][pth_path] = f"{module}.pth"

    def finalize(self, version, build_data, artifact_path):
        tmp = getattr(self, "_tmp", None)
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
