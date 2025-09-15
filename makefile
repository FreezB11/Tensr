CC = g++
VERSION = -std=c++23
INCLUDE = -I ./tensr/include/

test:
	${CC} ${VERSION} ${INCLUDE} ./tensr/src/*.cc main.cc -o we