#include <pthread.h>
#include <stdio.h>

void *thread(void *vargp);

int main() 
{
    pthread_t tid;

    printf("Hello World fron the main thread!\n");
    pthread_create(&tid, NULL, thread, NULL);
    pthread_join(tid, NULL);
    pthread_exit((void *)NULL);
}

void *thread(void *vargp) /* thread routine */
{
    printf("Hello World from a thread created by the main thread!\n"); 
    pthread_exit((void *)NULL);
}