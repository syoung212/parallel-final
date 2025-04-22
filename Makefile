CXX = g++
CXXFLAGS = -O3 -std=c++11 -Wall

SRC = main.cpp mcl.cpp
OBJ = main.o mcl.o
EXEC = mcl

all: $(EXEC)

main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c main.cpp -o main.o

mcl.o: mcl.cpp
	$(CXX) $(CXXFLAGS) -c mcl.cpp -o mcl.o

$(EXEC): $(OBJ)
	$(CXX) $(CXXFLAGS) $(OBJ) -o $(EXEC)

run: $(EXEC)
	@echo "Running with matrix file: $(matrix_file)"
	@./$(EXEC) $(matrix_file)

clean:
	rm -f $(OBJ) $(EXEC)

.PHONY: all run clean
