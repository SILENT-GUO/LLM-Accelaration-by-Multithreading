+ Report for Guo Zebin's PA2
+ UID: 3035770915
+ Note that for user time and system time, I use the data printed from main thread, as it was the **SUM** of all the child threads' user time and system time plus the main thread's user time and system time.

|Thread Numbers|Speed (tok/s)|User Time| System Time| Use Time/System Time|
|:---:|:---:|:---:|:---:|:---:|
0(Sequential)|53.090004|4.8108|0.1440|33.4083
1|42.838019|5.7963|0.2465|23.514
2|72.459666|5.9634|0.261|22.848
4|114.901257|6.3101|0.3136|20.121
6|134.524435|6.6715|0.5544|12.033
8|158.220025|6.8101|0.6504|10.4706
10|164.207826|7.5318|0.7231|10.4160
12|161.514196|8.1322|0.9558|8.5083
16|158.220025|10.0505|1.1443|8.7831

Analysis:
1. The speed of transformer increases as the number of threads increases generally. The advance of performance is the most significant when the thread number goes from 1 to 6. This is trivial and expected because multi-threading is designed to improve the speed of vector-matrix multiplication. 
2. However, another influence factor is the overhead of thread handling. We can see that when there are only one child thread handling the task, the speed is slower than the sequential version. This is because the overhead of thread handling is too significant to neglect. Also, when the number of threads increase from 10 to 16, the speed decreases. This is also because the overhead. The decrease of speed is not significant compared to the speed bolstering by multi-threading, but it is still worth noting. By analyzing the user time and system time, we can see that the overhead of thread handling mostly comes from the system time in the process of context switching.