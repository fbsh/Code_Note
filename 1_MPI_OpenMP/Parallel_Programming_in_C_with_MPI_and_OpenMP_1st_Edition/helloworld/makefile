EXECS = helloworld_MPI

SOURCE =  helloworld_MPI.c
MPICC = mpicc

all: $(SOURCE)
	$(MPICC)    $(SOURCE)  -o $(EXECS) 

clean:
	$(RM) -rf  $(EXECS) 

run:
	./$(EXECS)