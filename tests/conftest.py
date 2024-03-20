# conftest.py

def pytest_addoption(parser):
    # Grid parameters
    parser.addoption("--n-repeats", action="store", default=3, help="Number of Repeats")
    parser.addoption("--inv_dist", action="store_true", help="Use inv_dist Spacing")
