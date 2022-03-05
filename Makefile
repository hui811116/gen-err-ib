CXX=g++
CPPFLAGS= -std=c++11 -g


INCLUDE= -I ./Eigen/\
		 -I ./unsupported/

gen_bisearch: gen_bisearch.cpp
	@echo "Building $@"
	@${CXX} ${CPPFLAGS} ${INCLUDE} -o $@ $^

clean:
	@rm -f gen_bisearch