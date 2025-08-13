# Usage:
# make run FILE=main.cpp

CXX := g++
CXXFLAGS := -Wall -Wextra -std=c++17

NVCC := nvcc
# CXXFLAGS := -Wall -Wextra -std=c++17

SRC := $(word 2, $(MAKECMDGOALS))
BIN := $(basename $(notdir $(SRC)))

all:
	@echo "Usage: make run File=<program>.cpp"

$(BIN): $(SRC)
	@if [ "$(suffix $(SRC))" = ".cu" ]; then \
		echo "Compiling CUDA: $(SRC)"; \
		$(NVCC) $(SRC) -o build/$(BIN); \
	else \
		echo "Compiling C++: $(SRC)"; \
		$(CXX) $(CXXFLAGS) $(SRC) -o build/$(BIN); \
	fi

run: $(BIN)
	./build/$(BIN)

%:
	@:

clean:
	rm -f $(BIN)