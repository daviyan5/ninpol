# conftest.py

def pytest_addoption(parser):
    # Grid parameters
    parser.addoption("--n-repeats", action="store", default=1, help="Number of Repeats")
    parser.addoption("--n-files", action="store", default=-1, help="Number of Files")