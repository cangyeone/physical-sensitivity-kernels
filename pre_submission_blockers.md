# Pre-submission blockers for the GJI manuscript

These items must be resolved before formal journal submission. They are kept
outside the manuscript PDF so that the submitted source does not contain draft
or blocker language.

1. Mint a versioned public archive for the project.

   Required destination: Zenodo, OSF, or a GitHub release with an archived DOI.
   The public repository URL currently used in the manuscript is
   `https://github.com/cangyeone/SurfFlow`; a versioned DOI
   has not yet been minted for the manuscript-specific artifact bundle.

2. Upload large binary artifacts or provide approved checkpoint-access
   instructions.

   The manuscript tables can be reproduced from committed CSV/JSON outputs, but
   the trained checkpoint binaries are large local files and should be included
   in the public archive or accompanied by explicit download/regeneration
   instructions:

   - `ckpt/fair_di_strong_full_seed642026/best.pt`
   - `ckpt/fair_di_weak_full_seed642026/best.pt`
   - `ckpt/det_di_strong_full_seed642026/best.pt`
   - `ckpt/det_di_weak_full_seed642026/best.pt`
   - `ckpt/struct2disp_cpmlp.prior_boundary_v3.pt`

3. Confirm final archive metadata and license.

   The archive metadata should cite the manuscript title, authors, software
   license, dependency versions, random seed `642026`, and the Bayan Obo dataset
   DOI `10.5281/zenodo.17292491`. The dataset metadata used in the manuscript
   were checked against the Zenodo record before this revision.
