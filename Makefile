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
test:
	@echo "Testing..."
	@pytest -s

