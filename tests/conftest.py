# conftest.py

def pytest_addoption(parser):
    # Grid parameters
    parser.addoption("--n-repeats", action="store", default=3, help="Number of Repeats")
    parser.addoption("--linear", action="store_true", help="Use Linear Spacing")
