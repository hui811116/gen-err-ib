CXX=g++
CPPFLAGS= -std=c++11
BOOST_PATH=/Users/hui/project/tools

INCLUDE= -I ./Eigen/\
		 -I ./unsupported/\
		 -I $(BOOST_PATH)/include/

LIBRARY=$(BOOST_PATH)/lib
LIBS=$(LIBRARY)/libboost_program_options.a

gen_bisearch: gen_bisearch.cpp
	@echo "Building $@"
	@${CXX} ${CPPFLAGS} ${INCLUDE} -o $@ $^ ${LIBS}

clean:
	@rm -f gen_bisearch