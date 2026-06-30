"""Chemical family classification with mixed-anion separation."""

from pymatgen.core import Composition

_FAMILY_MAP = {}


def family_id(formula):
    if formula in _FAMILY_MAP:
        return _FAMILY_MAP[formula]
    comp = Composition(formula)
    els = {str(e) for e in comp.elements}
    has_hal = any(e in els for e in ["F", "Cl", "Br", "I"])
    has_o = "O" in els
    has_p = "P" in els
    has_s = "S" in els

    if has_hal and has_o:
        result = "oxyhalide"
    elif has_hal and has_s:
        result = "sulfohalide"
    elif has_hal:
        result = "pure_halide"
    elif has_o and has_p:
        result = "phosphate"
    elif has_o:
        result = "oxide"
    elif has_s:
        result = "sulfide"
    else:
        result = "other"

    _FAMILY_MAP[formula] = result
    return result


def family_numeric(formula):
    m = {
        "pure_halide": 0,
        "oxyhalide": 1,
        "sulfohalide": 2,
        "oxide": 3,
        "sulfide": 4,
        "phosphate": 5,
        "other": 6,
    }
    return m.get(family_id(formula), 6)


def has_lithium(formula):
    return "Li" in Composition(formula).element_composition
