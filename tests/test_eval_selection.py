"""Selection helpers for the multimer / protein-ligand runs (dev ``90`` §5.0a, P1 + P3).

The load-bearing assertions here are the two NEGATIVE ones: that ``_ca_array`` is unchanged, and that
the naive hetero filter would have been wrong. Both exist so a later "tidy-up" cannot quietly regress
either property.
"""

from __future__ import annotations

import numpy as np
import pytest

biotite_struc = pytest.importorskip("biotite.structure")

from spa.eval.score import _ca_array, adherence, polymer_only, select_chains  # noqa: E402


def _atom(chain, resid, resname, name, element, xyz, hetero=False):
    return biotite_struc.Atom(
        np.asarray(xyz, dtype="float32"),
        chain_id=chain, res_id=resid, res_name=resname,
        atom_name=name, element=element, hetero=hetero,
    )


def _mixed():
    """One ordinary residue, two MODIFIED residues, a calcium ion, and a ligand atom."""
    return biotite_struc.array([
        _atom("A", 1, "GLY", "CA", "C", [0.0, 0.0, 0.0]),
        _atom("A", 2, "MSE", "CA", "C", [3.8, 0.0, 0.0], hetero=True),
        _atom("A", 3, "SEP", "CA", "C", [7.6, 0.0, 0.0], hetero=True),
        _atom("Z", 901, "CA", "CA", "CA", [40.0, 0.0, 0.0], hetero=True),
        _atom("Z", 902, "IAI", "C1", "C", [41.0, 0.0, 0.0], hetero=True),
    ])


# ------------------------------------------------------------------- the bug these helpers exist for


def test_a_calcium_ion_is_counted_as_a_residue_by_the_unfiltered_selection():
    """The phantom residue is real: a Ca(2+) ion's atom NAME is literally 'CA'.

    This is the defect ``polymer_only`` exists to fix, asserted so it stays visible.
    """
    assert len(_ca_array(_mixed())) == 4                       # GLY, MSE, SEP + the ion
    assert "CA" in [str(a.res_name) for a in _ca_array(_mixed())]


def test_polymer_only_drops_the_ion_and_the_ligand_but_keeps_modified_residues():
    kept = [str(a.res_name) for a in _ca_array(polymer_only(_mixed()))]
    assert kept == ["GLY", "MSE", "SEP"]                       # ion and ligand gone, MSE/SEP kept


def test_the_naive_hetero_filter_would_have_deleted_real_residues():
    """Guard against a future 'simplification' to ``~arr.hetero``, which is measurably wrong.

    MSE and SEP are genuine residues with genuine Calpha atoms and are common in crystal structures.
    """
    arr = _mixed()
    naive = [str(a.res_name) for a in arr[~arr.hetero]]
    assert naive == ["GLY"]                                    # MSE and SEP wrongly deleted
    assert len(naive) < len([str(a.res_name) for a in _ca_array(polymer_only(arr))])


def test_ca_array_itself_is_untouched_by_the_new_helpers():
    """``_ca_array`` backs every historical number; it must keep its exact original behaviour."""
    arr = _mixed()
    assert len(_ca_array(arr)) == 4
    assert [str(a.res_name) for a in _ca_array(arr)] == ["GLY", "MSE", "SEP", "CA"]


def test_a_filtered_array_flows_through_the_existing_scorers_unmodified():
    """The whole design rests on AtomArrays passing through ``_as_struct``, so assert it."""
    a = adherence(polymer_only(_mixed()), polymer_only(_mixed()))
    assert a.n_design == a.n_prompt == 3
    assert a.prompt_rmsd == pytest.approx(0.0, abs=1e-6)


# ------------------------------------------------------------------------------------ select_chains


def _two_chain():
    atoms = [_atom("A", i + 1, "GLY", "CA", "C", [i * 3.8, 0.0, 0.0]) for i in range(5)]
    atoms += [_atom("B", i + 1, "GLY", "CA", "C", [i * 3.8, 10.0, 0.0]) for i in range(7)]
    return biotite_struc.array(atoms)


def test_select_chains_isolates_one_chain_so_the_scorers_stop_flattening():
    st = _two_chain()
    assert len(_ca_array(st)) == 12                            # today: both chains concatenated
    assert len(_ca_array(select_chains(st, "A"))) == 5
    assert len(_ca_array(select_chains(st, ["A", "B"]))) == 12


def test_select_chains_raises_on_an_absent_chain_rather_than_returning_nothing():
    with pytest.raises(ValueError, match="absent"):
        select_chains(_two_chain(), "Q")
