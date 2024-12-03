"""
Utility method which prints system info to help with debugging,
and filing issues on GitHub.
Adapted from :func:`sklearn.show_versions`,
which was adapted from :func:`pandas.show_versions`
"""
import sys
from .. import __version__

def _get_deps_info():
    """Overview of the installed version of main dependencies
    Returns
    -------
    deps_info: dict
        version information on relevant Python libraries
    """
    import importlib
    deps_info = {}
    deps = [
        "pip",
        "setuptools",
        "sklearn",
        "numpy",
        "scipy",
        "pandas",
        "tensorflow",
        "keras",
    ]
    for modname in deps:
        try:
            if modname in sys.modules:
                mod = sys.modules[modname]
            else:
                mod = importlib.import_module(modname)
            ver = getattr(mod, "__version__", "unknown")
            deps_info[modname] = ver
        except ImportError:
            deps_info[modname] = None
    return deps_info

def show_versions(github=False):
    """Print debugging information.

    .. versionadded:: 0.5

    Parameters
    ----------
    github : bool,
        If true, wrap system info with GitHub markup.
    """
    import sys
    import platform

    sys_info = {
        "python": sys.version,
        "executable": sys.executable,
        "machine": platform.machine(),
        "platform": platform.platform(),
    }

    deps_info = _get_deps_info()

    if github:
        sys_info = {f"* {k}: `{v}`" for k, v in sys_info.items()}
        deps_info = {f"* {k}: `{v}`" for k, v in deps_info.items()}

    print("\nSystem:")
    print("\n".join(f"{k}: {v}" for k, v in sys_info.items()))

    print("\nPython dependencies:")
    print(f"* imbalanced-learn: {__version__}")
    print("\n".join(f"* {k}: {v}" for k, v in deps_info.items()))

    if github:
        print("\nCopy and paste the above information in a GitHub issue.")
