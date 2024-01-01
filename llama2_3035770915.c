/*
PLEASE WRITE DOWN FOLLOWING INFO BEFORE SUBMISSION
* FILE NAME: llama_3035770915.c
* NAME: Guo Zebin
* UID :3035770915
* Development Platform: workbench2
* Remark: (How much you implemented?) All in the critiria
* How to compile: (  gcc -o llama2_3035770915 llama2_3035770915.c utilities.c -O2 -pthread -lm)
* The seed used for report is 42. You can reporduce it by running: ./llama2_3035770915 42 <thr_count>
Please download the model and tokenizer to the same folder:
$ wget -O model.bin https://huggingface.co/huangs0/llama2.c/resolve/main/model.bin
$ wget -O tokenizer.bin https://huggingface.co/huangs0/llama2.c/resolve/main/tokenizer.bin

In compile, remember to add `-pthred` to link library:
$ gcc -o llama2_[UID] llama2_[UID].c utilities.c -O2 -pthread -lm

Then Run with:
$ ./llama2_[UID] <seed> <thr_count>
*/

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <sys/resource.h>
#include "utilities.h"

/**
 * ----------------------------------------------------------------------------
 * TASK - Optimize Matrix-Vector Multiplication by Multi-Threading
 *
 * Matrix-Vector Multiplication, used in Attention and Feed-Forward Network
 * is the most time-consuming part of GPT. Luckily, most of computation is
 * independent of each row, so we can use Multi-Threading for acceleration.
 *
 * Please use <pthread.h> and your favorite synchronization method,
 * semaphore / mutex lock + conditional variable
 *
 * A sequential version is provided in seq.c, please modify it to parallel version.
 */

// YOUR CODE STARTS HERE

// Addtional Header File Here
#include <pthread.h>
#include <semaphore.h>
// Global Variables
struct rusage main_usage; // get usage for main thread
typedef struct
{
    int thread_id; // index of thread
    float *out; //same copy of params in mat_vec_mul function
    float *vec;
    float *mat;
    int col;
    int row;
    int end;
} Thread_params;

int n = -1;            // number of threads
//semaphore array for each thread; 
//sem_thread_executing records the number of threads executing, used for checking if all threads are finished
sem_t *semaphore_array = NULL, sem_thread_executing;
//thread array is used for pthread_create and pthread_join
//thread_params_array is used for passing params to threads
pthread_t *thread_array = NULL;
Thread_params *thread_params_array = NULL;
//sem_mutex_lock is used for locking the main thread when call system usage
sem_t sem_mutex_lock;

void *thr_func(void *arg);

int init_mat_vec_mul(int thr_count)
{
    semaphore_array = malloc(thr_count * sizeof(sem_t));
    thread_array = malloc(thr_count * sizeof(pthread_t));
    thread_params_array = (Thread_params *)malloc(thr_count * sizeof(Thread_params));

    sem_init(&sem_mutex_lock, 0, 1);
    sem_init(&sem_thread_executing, 0, 0);
    n = thr_count;
    // init n threads
    for (int i = 0; i < n; i++)
    {
        if (sem_init(&semaphore_array[i], 0, 0) != 0)
        {
            perror("sem_init");
            exit(EXIT_FAILURE);
        }
        //init thread params
        thread_params_array[i].thread_id = i;
        thread_params_array[i].out = NULL;
        thread_params_array[i].vec = NULL;
        thread_params_array[i].mat = NULL;
        thread_params_array[i].col = 0;
        thread_params_array[i].row = 0;
        thread_params_array[i].end = 0;

        if (pthread_create(&thread_array[i], NULL, thr_func, &thread_params_array[i]) != 0)
        {
            perror("pthread_create");
            exit(EXIT_FAILURE);
        }
    }
    return 0;
}

void mat_vec_mul(float *out, float *vec, float *mat, int col, int row)
{

    // out, vec, mat, col, row are pre-allocated
    for (int i = 0; i < n; i++)
    {
        thread_params_array[i].out = out;
        thread_params_array[i].vec = vec;
        thread_params_array[i].mat = mat;
        thread_params_array[i].col = col;
        thread_params_array[i].row = row;
    }

    // iterate through all threads
    for (int j = 0; j < n; j++)
    {
        // wake up thread
        sem_post(&semaphore_array[j]);
    }

    // wait for all threads to finish
    for (int j = 0; j < n; j++)
    {
        sem_wait(&sem_thread_executing);
    }
}

int close_mat_vec_mul()
{
    // wake up all threads, and set end to 1 to notify threads to exit
    for (int i = 0; i < n; i++)
    {
        thread_params_array[i].end = 1;
        sem_post(&semaphore_array[i]);
    }
    // wait for all threads to exit
    for (int i = 0; i < n; i++)
    {
        if (pthread_join(thread_array[i], NULL) != 0)
        {
            perror("pthread_join");
            exit(EXIT_FAILURE);
        }
    }
    // collect system usage of the main thread

    if (getrusage(RUSAGE_SELF, &main_usage) != 0)
    {
        perror("getrusage");
        exit(EXIT_FAILURE);
    }
    // print system usage of the main thread
    printf("main thread - user: %.4f s, system: %.4f s\n",
           (main_usage.ru_utime.tv_sec + main_usage.ru_utime.tv_usec / 1000000.0),
           (main_usage.ru_stime.tv_sec + main_usage.ru_stime.tv_usec / 1000000.0));
    // free memory
    free(semaphore_array);
    free(thread_array);
    free(thread_params_array);

    sem_destroy(&sem_mutex_lock);
    sem_destroy(&sem_thread_executing);
    

    return 0;
}

void *thr_func(void *arg)
{

    while (1)
    {
        // wait for main thread to wake up
        int thread_id = ((Thread_params *)arg)->thread_id;
        sem_wait(&semaphore_array[thread_id]);
        // get params
        int end_thread = ((Thread_params *)arg)->end;
        float *out = ((Thread_params *)arg)->out;
        float *vec = ((Thread_params *)arg)->vec;
        float *mat = ((Thread_params *)arg)->mat;
        int col = ((Thread_params *)arg)->col;
        int row = ((Thread_params *)arg)->row;
        if (end_thread == 1)
        {
            sem_wait(&sem_mutex_lock);
            if (getrusage(RUSAGE_THREAD, &main_usage) != 0)
            {
                perror("getrusage");
                exit(EXIT_FAILURE);
            }
            printf("Thread %d has completed - user: %.4f s, system: %.4f s\n", thread_id,
                   (main_usage.ru_utime.tv_sec + main_usage.ru_utime.tv_usec / 1000000.0),
                   (main_usage.ru_stime.tv_sec + main_usage.ru_stime.tv_usec / 1000000.0));
            sem_post(&sem_mutex_lock);
            break;
        }
        // check if params are valid
        if (out == NULL || vec == NULL || mat == NULL || col == 0 || row == 0)
        { // error case
            perror("thread error");
            exit(EXIT_FAILURE);
        }
        // calculate scope of each thread
        int scope = (row + (n - 1)) / n;
        int start = thread_id * scope;
        int end = (thread_id + 1) * scope;
        if (end > row)
        {
            end = row;
        }
        // calculate matrix-vector multiplication
        for (int i = start; i < end; i++)
        {
            out[i] = 0;
            for (int j = 0; j < col; j++)
            {
                out[i] += vec[j] * mat[i * col + j];
            }
        }
        // notify main thread that this thread has finished current task and wait for next task
        sem_post(&sem_thread_executing);
    }
}

// YOUR CODE ENDS HERE

int transformer(int token, int pos, LLMConfig *p, LLMRuntime *s, LLMWeight *w)
{

    // a few convenience variables
    int dim = p->dim, hidden_dim = p->hidden_dim, head_size = p->dim / p->n_heads;

    // copy the token embedding into x
    memcpy(s->x, &(w->token_embedding_table[token * dim]), dim * sizeof(float));

    // forward all the layers
    for (int l = 0; l < p->n_layers; l++)
    {

        // Attention
        {
            // attention normalization
            normalize(s->xb, s->x, w->rms_att_weight + l * dim, dim);

            // q, k, v = w_q @ x, w_k @ x, w_v @ x, respectively
            mat_vec_mul(s->q, s->xb, w->wq + l * dim * dim, dim, dim);
            mat_vec_mul(s->k, s->xb, w->wk + l * dim * dim, dim, dim);
            mat_vec_mul(s->v, s->xb, w->wv + l * dim * dim, dim, dim);

            // apply positional embedding
            position_embedding(s->q, s->k, w, pos, p->dim, p->n_heads);

            // save intermediate result for later reference
            key_value_cache(l, pos, p, s);

            // attention calculation
            attention(l, pos, p, s, w);

            // wo @ x to get final result
            mat_vec_mul(s->xb2, s->xb, w->wo + l * dim * dim, dim, dim);

            // residual connection back into x
            accum(s->x, s->xb2, dim);
        }

        // Feed-Forward Network: w2 @ (silu(w1 @ x) * (w3 @ x)), * is element-wise multiply
        {
            // FFN Normalization
            normalize(s->xb, s->x, w->rms_ffn_weight + l * dim, dim);

            // w1 @ x
            mat_vec_mul(s->h1, s->xb, w->w1 + l * dim * hidden_dim, dim, hidden_dim);
            // silu(w1 @ x)
            silu(s->h1, hidden_dim);
            // w3 @ x
            mat_vec_mul(s->h2, s->xb, w->w3 + l * dim * hidden_dim, dim, hidden_dim);
            // silu(w1 @ x) * (w3 @ x)
            element_wise_mul(s->h1, s->h2, hidden_dim);
            // w2 @ (silu(w1 @ x) * (w3 @ x))
            mat_vec_mul(s->xb, s->h1, w->w2 + l * dim * hidden_dim, hidden_dim, dim);

            // residual connection
            accum(s->x, s->xb, dim);
        }
    }

    // final normalization
    normalize(s->x, s->x, w->rms_final_weight, dim);
    // classifier into logits
    mat_vec_mul(s->logits, s->x, w->token_embedding_table, p->dim, p->vocab_size);
    // apply the temperature to the logits
    for (int q = 0; q < p->vocab_size; q++)
    {
        s->logits[q] /= 0.9f;
    }
    // apply softmax to the logits to get the probabilities for next token
    softmax(s->logits, p->vocab_size);
    // now sample from this distribution to get the next token
    return sample(s->logits, p->vocab_size);
}

int main(int argc, char *argv[])
{

    unsigned int seed;
    int thr_count;

    if (argc == 3)
    {
        seed = atoi(argv[1]);
        thr_count = atoi(argv[2]);
    }
    else
    {
        printf("Usage: ./compiled <seed> <thr_count>\n");
        return 1;
    }

    // Initialize
    srand(seed);
    init_mat_vec_mul(thr_count);

    // load model
    LLMConfig config;
    LLMWeight weights;
    if (load_LLM_Config_Weight(&config, &weights) == 1)
    {
        return 1;
    }

    // load tokenizer
    char **vocab = malloc(config.vocab_size * sizeof(char *));
    if (load_tokenizer(vocab, config.vocab_size) == 1)
    {
        return 1;
    }

    // create and init the application LLMRuntime
    LLMRuntime state;
    malloc_LLMRuntime(&state, &config);

    // the current position we are in
    long start = time_in_ms();

    int next, token = 1, pos = 0; // token = 1 -> <START>
    while (pos < config.seq_len)
    {

        // forward the transformer to get logits for the next token
        next = transformer(token, pos, &config, &state, &weights);

        printf("%s", vocab[next]);
        fflush(stdout); // force print

        token = next;
        pos++;
    }

    long end = time_in_ms();
    printf("\n\nlength: %d, time: %f s, achieved tok/s: %f\n", config.seq_len, (double)(end - start) / 1000, config.seq_len / (double)(end - start) * 1000);

    // cleanup
    close_mat_vec_mul();
    free_LLMRuntime(&state);
    free_LLMWeight(&weights);
    for (int i = 0; i < config.vocab_size; i++)
    {
        free(vocab[i]);
    }
    free(vocab);
    return 0;
}