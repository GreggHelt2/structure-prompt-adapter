"""Tier-1 structural metrics: ligand pocket + inter-chain interface (dev ``90`` §5.0a P4/P5).

Both are refold-free geometry on the generated backbone. These tests build tiny synthetic AtomArrays
rather than reading fixtures, so they run anywhere and assert exact, hand-checkable numbers.
"""

from __future__ import annotations

import numpy as np
import pytest

biotite_struc = pytest.importorskip("biotite.structure")

from spa.eval.score import ligand_pocket, interface_geometry  # noqa: E402


def _atom(chain, resid, resname, name, element, xyz, hetero=False):
    return biotite_struc.Atom(
        np.asarray(xyz, dtype="float32"),
        chain_id=chain, res_id=resid, res_name=resname,
        atom_name=name, element=element, hetero=hetero,
    )


def _residue(chain, resid, x, resname="GLY"):
    """One glycine's N, CA, C at x, x+1, x+2 along the x axis."""
    return [
        _atom(chain, resid, resname, "N", "N", [x, 0.0, 0.0]),
        _atom(chain, resid, resname, "CA", "C", [x + 1.0, 0.0, 0.0]),
        _atom(chain, resid, resname, "C", "C", [x + 2.0, 0.0, 0.0]),
    ]


def _array(atoms):
    return biotite_struc.array(atoms)


# ----------------------------------------------------------------------------------- ligand pocket


def _with_ligand(protein_x, lig_xyz=(0.0, 10.0, 0.0)):
    atoms = _residue("A", 1, protein_x) + [
        _atom("Z", 900, "IAI", "C1", "C", lig_xyz, hetero=True)
    ]
    return _array(atoms)


def test_ligand_pocket_counts_clash_at_rfd3_cutoff():
    """Backbone within 1.5 A of the ligand clashes, per RFD3's AME definition; beyond it does not.

    The residue puts N/CA/C at x = 0/1/2. With the ligand at (2, 1.4, 0) only C is inside 1.5 A
    (1.400); CA is 1.720 and N is 2.441. Moving the ligand to y = 2 leaves nothing inside.
    """
    close = _array(_residue("A", 1, 0.0) + [_atom("Z", 900, "IAI", "C1", "C", [2.0, 1.4, 0.0], True)])
    lp = ligand_pocket(close, "IAI")
    assert lp.n_clash == 1
    assert lp.min_dist == pytest.approx(1.4, abs=1e-5)

    far = _array(_residue("A", 1, 0.0) + [_atom("Z", 900, "IAI", "C1", "C", [2.0, 2.0, 0.0], True)])
    assert ligand_pocket(far, "IAI").n_clash == 0            # nearest backbone atom is 2.0 A


def test_ligand_pocket_excludes_motif_residues_from_the_clash_count():
    """RFD3's definition counts NON-motif backbone atoms only, so a pinned residue never clashes."""
    st = _array(_residue("A", 1, 0.0) + [_atom("Z", 900, "IAI", "C1", "C", [2.0, 1.4, 0.0], True)])
    assert ligand_pocket(st, "IAI").n_clash == 1
    assert ligand_pocket(st, "IAI", motif_atoms=[("A", 1, "CA")]).n_clash == 0


def test_ligand_pocket_motif_distance_and_rmsd_against_a_reference():
    design = _with_ligand(0.0)                                # CA at (1,0,0), ligand at (0,10,0)
    ref = _with_ligand(0.0)
    lp = ligand_pocket(design, "IAI", motif_atoms=[("A", 1, "CA")], reference=ref)
    assert lp.motif_dists[0] == pytest.approx(np.hypot(1.0, 10.0))
    assert lp.motif_dist_rmsd == pytest.approx(0.0)           # identical -> pocket fully preserved
    assert lp.n_ligand_atoms == 1

    moved = _with_ligand(3.0)                                 # CA now at (4,0,0)
    lp2 = ligand_pocket(moved, "IAI", motif_atoms=[("A", 1, "CA")], reference=ref)
    assert lp2.motif_dist_rmsd > 0.5                          # displaced -> pocket degraded


def test_ligand_pocket_raises_on_absent_ligand_rather_than_scoring_zero():
    st = _array(_residue("A", 1, 0.0))
    with pytest.raises(ValueError, match="absent"):
        ligand_pocket(st, "IAI")


# --------------------------------------------------------------------------------------- interface


def _complex(a_offset=0.0, n=4, gap=5.0):
    """Two 4-residue chains, A along y=0 and B along y=gap, both stepping in x."""
    atoms = []
    for i in range(n):
        atoms += _residue("A", i + 1, i * 4.0 + a_offset)
    for i in range(n):
        b = _residue("B", i + 1, i * 4.0)
        atoms += [
            biotite_struc.Atom(
                np.asarray([at.coord[0], gap, 0.0], dtype="float32"),
                chain_id=at.chain_id, res_id=at.res_id, res_name=at.res_name,
                atom_name=at.atom_name, element=at.element, hetero=at.hetero,
            ) for at in b
        ]
    return _array(atoms)


def test_interface_counts_cross_chain_contacts_only():
    st = _complex()
    iface = interface_geometry(st, "A", "B", cutoff=8.0)
    assert iface.n_contacts > 0
    assert iface.n_interface_a > 0 and iface.n_interface_b > 0
    # A tight cutoff below the 5 A chain separation must find nothing.
    assert interface_geometry(st, "A", "B", cutoff=4.0).n_contacts == 0


def test_interface_retention_and_rmsd_are_zero_change_for_an_identical_reference():
    st = _complex()
    iface = interface_geometry(st, "A", "B", reference=_complex())
    assert iface.frac_retained == pytest.approx(1.0)
    assert iface.rmsd == pytest.approx(0.0, abs=1e-6)


def test_interface_degrades_when_the_steered_chain_slides():
    ref = _complex()
    moved = _complex(a_offset=6.0)
    iface = interface_geometry(moved, "A", "B", reference=ref)
    assert iface.rmsd is not None and iface.rmsd > 1.0


def test_interface_raises_when_the_partner_chain_is_not_held_fixed():
    """A partner of a different length means it was diffused, which invalidates the superposition."""
    with pytest.raises(ValueError, match="held fixed"):
        interface_geometry(_complex(n=4), "A", "B", reference=_complex(n=5))


def test_interface_raises_on_a_missing_chain():
    with pytest.raises(ValueError, match="present chains"):
        interface_geometry(_complex(), "A", "Q")
