CC = g++
CFLAGS = -std=c++11 -O3 -Wall
LDFLAGS = -lpthread -lm -lopencv_world -lapriltag -lcurl -lwiringPi

all: calibrate_cameras.x locate_cameras.x locate_tags.x crop.x

%.x: %.cc
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

.PHONY: all

clean:
	-rm *.x
