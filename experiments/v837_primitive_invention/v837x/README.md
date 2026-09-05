# V837x — Minimal Global Scalar Controller Transfer

Authorized only by V837w's `JOINT_INPUT_STATE_GLOBAL_CONTROL_REQUIRED` result. X2 computes one scalar from the concatenated 40D previous neutral state plus current 6D observation, once at the start of each timestep, and broadcasts it to all ten cells. X2C uses the same controller and parameters but removes old-state carry.
