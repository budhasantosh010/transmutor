# V837p — Minimal Generic Dynamic State Modulator

V837p is the single neutral follow-up licensed by V837o.

V837o established that every static update/reset combination remained at 3/5 while either dynamic update or dynamic reset retained 5/5. Therefore V837p does **not** copy a GRU gate. It tests the smallest shared property supported by that result: one generic dynamic scalar coefficient per neutral cell, conditioned on the cell state, aggregate message, and raw input.

The coefficient modulates recurrent-state access before the historical tanh candidate. The exact parameter-matched control uses the same dynamic scalar network but adds the coefficient to the candidate preactivation instead of multiplying the previous state.

All conditions use the fixed high-capacity neutral topology, 4× unique development data, 192 optimizer steps, paired seeds, and the original five V837 task generators. Fresh-audit seeds and primitive mining remain locked.
