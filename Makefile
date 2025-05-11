CXX      = g++
CXXFLAGS = -O3 -std=c++11 -Wall
OMPFLAGS = -fopenmp

SRC      = main.cpp \
           mcl_serial.cpp \
           mcl_openmp.cpp \
		   mcl_row.cpp
OBJ      = $(SRC:.cpp=.o)
EXEC     = mcl_all

all: $(EXEC)

$(EXEC): $(OBJ)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $^ -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(if $(findstring openmp,$<),$(OMPFLAGS),) -c $< -o $@

clean:
	rm -f $(OBJ) $(EXEC)

.PHONY: all clean
