# Makefile for Markov Clustering on Perlmutter

CXX = g++
CXXFLAGS = -O3 -std=c++11 -Wall

TARGET = mcl
SRC = mcl.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)
