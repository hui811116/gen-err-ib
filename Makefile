CXX=g++
CPPFLAGS= -std=c++11
BOOST_PATH=/Users/hui/project/tools

INCLUDE= -I ./Eigen/\
		 -I ./unsupported/\
		 -I $(BOOST_PATH)/include/

LIBRARY=$(BOOST_PATH)/lib
LIBS=$(LIBRARY)/libboost_program_options.a

targets = gen_bisearch gen_samp_adversary
#.PHONY: all
all: $(targets)

#$(targets): .cpp
#	@echo "Building $@"
#	@${CXX} ${CPPFLAGS} ${INCLUDE} -o $@ $^ ${LIBS}

gen_bisearch: gen_bisearch.cpp
	@echo "Building $@"
	@${CXX} ${CPPFLAGS} ${INCLUDE} -o $@ $^ ${LIBS}

gen_samp_adversary: gen_samp_adversary.cpp
	@echo "Building $@"
	@${CXX} ${CPPFLAGS} ${INCLUDE} -o $@ $^ ${LIBS}



clean:
	@rm -f $(targets)