# V837al frozen research specification

Question: **What is the smallest explicit transformation of a primitive's boundary interface that makes independently learned instances of the same computational class interoperable?**

Frozen source: V837ak at repository start SHA `802478151a5ab0d4cbbe099e3c18f056e8ce65bd`.

Data roles: ALIGN_FIT development 10000–10063; ALIGN_SELECT development 10064–10127; ALIGN_TEST validation 20000–20127. ALIGN_TEST is held out from adapter fitting/selection but was historically involved in V837ak motif science; that limitation is explicit.

Ports: recurrent state, external message contribution, global coupling contribution, projected candidate input, global scalar gate, outgoing local output.

Families: signed permutation, diagonal affine, rigid affine, full affine. All fits are deterministic closed form; no optimizer is permitted.

All 64 port scopes are enumerated, producing identity plus 63×4 = 253 configurations. GLOBAL and PRIMARY_CAUSAL tracks are predeclared. Test failure cannot trigger a retry. Pairwise evidence does not authorize an archive; canonical plus causal closed-loop evidence is required.
