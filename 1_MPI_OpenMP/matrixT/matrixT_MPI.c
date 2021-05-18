#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <mpi.h>

//#define INDEX(r,c,i,j) r*c + j
//#define ACCESS(A,i,j) A->arr[INDEX(A->rows, A->cols, i, j)]

#define INDEX(r,c,i,j) i*r + j

struct matrix {
  int rows;
  int cols;
  int *arr;
};

void initMatrix(struct matrix *A, int r, int c){
  A->rows = r;
  A->cols = c;
  int size = (r*c)*sizeof(int);
  A->arr = malloc(size);

  int i, j;
  for(i = 0; i < r; i++){
    for(j = 0; j < c; j++){
      int index = INDEX(r,c, i, j);
      A->arr[index] = rand() % 100 + 1;
    }
  }
}

void printMatrix(struct matrix *A){

  printf("\n");
  int i, j;
  for(i = 0; i < A->rows; i++){
    for(j = 0; j < A->cols; j++){
      int index = INDEX(A->rows,A->cols, i, j);//(i*A->rows) + j;
      printf("%d ", A->arr[index]);
    }
    puts("");
  }
  printf("\n");
}

struct matrix matrixTranspose(struct matrix *A){

  //strange bug: would not overwrite A in this function
  //applying to temporary fixes this

  struct matrix B;
  initMatrix(&B, A->rows, A->cols);

  int i, j;
  for(i = 0; i < A->rows; i++){
    for(j = 0; j < A->cols; j++){

      int index = INDEX(A->rows, A->cols, i, j);
      int newIndex = INDEX(A->rows, A->cols, j, i);

      int first = A->arr[index];
      int second = A->arr[newIndex];
      B.arr[index] = second;
      B.arr[newIndex] = first;

    }

  }

  return B;

}

int main(int argc, char **argv){

  int wsize, rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &wsize);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  //have root node init matrix

  if(rank == 0){

    //As mentioned in class (September 17)
    //do not parallelize transpose operation

    int num;
    sscanf(argv[1],"%d",&num);
    int msize = num;

    srand(time(0));
    struct matrix A;
    initMatrix(&A, msize, msize);
    //printMatrix(&A);

    int i;
    for(i = 0; i < (A.rows*A.cols); i++){

      //printf("%d: %d, \n", i, A.arr[i]);
    }

    struct matrix B = matrixTranspose(&A);
    printf("\n");
    //printMatrix(&B);

  }

  MPI_Finalize();

  return 0;
}
