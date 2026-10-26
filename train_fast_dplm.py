"""Bounded debug entrypoint; GPU execution belongs in Slurm."""
from fast_dplm.training import main

if __name__ == "__main__":
    main(debug=True)
