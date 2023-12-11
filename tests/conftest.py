# conftest.py

def pytest_addoption(parser):
    # Grid parameters
    parser.addoption("--min-elem", action="store", default=5, help="Minimum number of elements")
    parser.addoption("--max-elem", action="store", default=30000, help="Maximum number of elements")
    parser.addoption("--n-test", action="store", default=6, help="Number of tests")
    parser.addoption("--n-repeats", action="store", default=3, help="Number of repeats")
    parser.addoption("--linear", action="store_true", help="Use linear spacing")
