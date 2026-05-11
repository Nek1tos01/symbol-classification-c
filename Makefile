CC=gcc
CFLAGS=-std=c11 -O2 -Wall -Wextra -pedantic -Iinclude
LDFLAGS=-lm
TARGET=nn_classifier
SRC=$(wildcard src/*.c)
OBJ=$(SRC:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(OBJ) -o $@ $(LDFLAGS)

src/%.o: src/%.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	del /Q src\*.o $(TARGET).exe 2>nul || exit 0

run: $(TARGET)
	./$(TARGET).exe config.txt

.PHONY: all clean run
