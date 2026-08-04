# test_prediction.py is the L4 analysis module, not a pytest suite — its
# test_forward_prediction* functions take (df, sector_prefix, forward_days)
# and pytest reads those params as missing fixtures. test_components.py is
# the only real test suite here.
collect_ignore = ["test_prediction.py"]
