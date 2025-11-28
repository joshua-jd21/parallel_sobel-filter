CC = gcc
CFLAGS = -O2 -fopenmp -Wall
TARGET = sobel_omp
SRC = sobel_omp.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) -lm

clean:
	rm -f $(TARGET) *.o

.PHONY: all clean
