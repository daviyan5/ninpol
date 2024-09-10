# Runs:
# 	python3 setup.py build_ext --inplace && python3 pip install .
all:
	make install

install: compile
	@echo "Installing..."
	@python3 -m pip install .
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
	@python3 -m pip install .
	@echo "Done"
compile_debug:
	@echo "Compilling in debug mode..."
	@python3 setup.py build_ext --inplace --debug
	@echo "Done"
test:
	@echo "Testing..."
	@pytest -s --tb=short