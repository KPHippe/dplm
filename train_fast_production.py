"""Frozen-config production entrypoint. Submit larger runs only after user review."""
from fast_dplm.training import main

if __name__ == '__main__':
    main(debug=False)
