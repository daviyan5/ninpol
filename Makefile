# Runs:
# 	python3 setup.py build_ext --inplace && python3 pip install .
all:
	make install

install: compile
	@echo "Installing..."
	@pip install .
	@echo "Done"
compile:
	@echo "Compilling..."
	@python3 setup.py build_ext --inplace
	@echo "Done" 
clean:
	@echo "Cleaning..."
	@python3 build/clean.py
	@echo "Done"
debug: compile_debug
	@echo "Installing..."
	@pip install .
	@echo "Done"
compile_debug:
	@echo "Compilling in debug mode..."
	@python3 setup.py build_ext --inplace --debug
	@echo "Done"
create:
	@echo "Creating vtk files..."
	@python3 tests/utils/create_vtk.py
test_create: create
	@echo "Testing..."
	@pytest -s --tb=short

	@echo "Running memory tests..."
	@valgrind --tool=memcheck --suppressions=tests/utils/valgrind-python.supp python tests/utils/try_one.py
test:
	@echo "Testing..."
	@pytest -s --tb=short

	@echo "Running memory tests..."
	@valgrind --tool=memcheck --suppressions=tests/utils/valgrind-python.supp python tests/utils/try_one.py
