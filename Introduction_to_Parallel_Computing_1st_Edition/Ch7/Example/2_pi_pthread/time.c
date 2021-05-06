#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <stdio.h>

double get_time()
{
    struct timeval t;
    struct timezone tzp;
    gettimeofday(&t, &tzp);
    return t.tv_sec + t.tv_usec*1e-6;
}

int main(){
    double i = get_time();
    printf("%f\n", i);
    return 0;
}