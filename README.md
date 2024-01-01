# LLM-Accelaration-by-Multithreading
Author: Guo Zebin

This project accelerate LLM inferencing by using multi-threading in matrix-vector multiplications when using transformer.
This is the second programming assignment of HKU COMP3230 course.

## How to run it:
Platform: Linux (workbench2 in school server)

Compile the code: 
```
gcc -o llama2_3035770915 llama2_3035770915.c utilities.c -O2 -pthread -lm
```
Run the code:
```
./llama2_3035770915 42 <thr_count>
```
+ 42 is the random seed for text generation, different seeds correspond to different text, you can choose any integer between 1 to 100.
+ thr_count is the number of thread. When thread number is low, increasing the thread number will result in faster generation speed, but the thread handling overhead may forestall the accelaration when the thread number is high. Please refer to report_3035770915.pdf for detailed thread number comparison.

