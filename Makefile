# Runs:
# 	python3 setup.py build_ext --inplace && python3 pip install .
all:
	make install
	make clean
install: compile
	@echo "Installing..."
	@pip install .
	@echo "Done"
compile: list
	@echo "Compilling..."
	@python3 setup.py build_ext --inplace
	@echo "Done" 
clean:
	@echo "Cleaning..."
	@python3 build-helpers/clean.py
	@rm -rf build
	@rm -rf interpolator.egg-info
	@echo "Done"
list:
	@echo "Listing..."
	@python3 build-helpers/list.py
	@echo "Done"
test:
	@echo "Testing..."

