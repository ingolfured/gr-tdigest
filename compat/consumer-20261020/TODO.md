# Pending source and evidence comparison

Requested by Ingo, 20 October 2026. Keep pending until explicitly compared.

- [ ] Read `/run/media/ingo/Storage/git/gr-tdigest/README.md` and `AGENTS.md` before operating on that checkout.
- [ ] Identify exact external SSD HEAD, cached remote ref, internal checkout HEAD and ancestry; confirm the reported shared HEAD object without assuming ancestry or source equivalence.
- [ ] Compare source and original experiment material; preserve historical branches and all useful evidence until this comparison is complete.
- [ ] Assess the large SSD `target` directory separately as generated build cache. Do not use cache cleanup as authorization to remove source, branches or measured artifacts.

Reported context (not newly verified): SSD checkout was clean and matched a locally cached remote ref; its HEAD object existed internally but was not an ancestor of the internal checkout HEAD at inspection time.
