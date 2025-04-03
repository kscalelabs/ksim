"""Teacher-student training for default humanoid walking task.

We take a trained teacher policy and distill it into an LSTM-based student
policy with a KL-divergence loss, using lots of domain randomization to
help it transfer to the real world effectively.
"""
