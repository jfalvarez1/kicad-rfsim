# Multiport wave extraction

The previous openEMS runner formed each S-column from `b_j / a_k`. That
requires all other incident waves to vanish. Renormalizing an MSL port to a
real reference impedance does not guarantee that condition. The runner now
retains the incident and reflected waves at every port in every excitation
and solves **B = S A**, frequency by frequency. The field-solver geometry,
materials and mesh are unchanged.

`port_waves.npz` preserves the complex raw waves and frequencies.
`sparameter-quality.json` records conditioning, reciprocity error and the
largest singular value of S. With equal real reference impedances, passive
networks require that value to be at most 1. Checking the power of individual
columns alone is insufficient for coherent excitation of several ports.
See [Ansys's passivity definition](https://ansyshelp.ansys.com/public/Views/Secured/Electronics/v242/en/Subsystems/Circuit/Passivity.htm).

The runner rejects nonfinite or ill-conditioned wave matrices. Partial
excitation runs retain their waves but do not export a complete Touchstone
network with invented zero columns. Select all excitation ports for export.
Quality warnings do not alter or clip the calculated S-parameters.

## Reproduce

```console
python validation/common/test_wave_matrix.py
python validation/common/test_touchstone.py
python validation/common/check_port_waves.py validation/data/microstrip-multiport/port_waves.npz
```

The checked-in fixture was run with openEMS 0.37.0 / CSXCAD 0.7.0, a 1–6 GHz
sweep with 401 points, and the medium mesh. It is a synthetic microstrip,
not a measurement or an arbitrary autorouted board. Its exact extracted
geometry and material assumptions are in `model.json` next to the waves.
The same waves can be evaluated by both extraction methods, isolating the
postprocessing change from the field solve. `comparison.json` contains that
comparison; `results.s2p` is the corrected network.

Coarse, medium and fine fresh runs produced maximum singular values of
0.998086, 0.997371 and 0.997067 respectively. These pass a necessary physical
check. They do not establish mesh convergence, accuracy against hardware,
causality, or correctness for all port types. Further validation should use
measured coupons and fixtures with CPW, stripline and lumped ports.

The patch and validation scripts were generated with Codex and tested locally.
