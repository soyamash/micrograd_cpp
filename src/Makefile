CC = g++
CFLAGS = -O3 -Wall -std=c++11 -g
SRCS = main.cpp engine.cpp nn.cpp
LIBS = 
INCDIR = 
OBJS = $(SRCS:.cpp=.o)

all: main

$(OBJS): $(SRCS)
	$(CC) $(CFLAGS) $(INCDIR) -c $(SRCS)

main: $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) $(LIBS) -o main

clean:
	rm -f *.o *~ main