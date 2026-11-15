# Evaluation

CPU hardware: {'platform': 'Linux-6.6.159+-x86_64-with-glibc2.39', 'processor': 'x86_64', 'machine': 'x86_64', 'logical_cpus': 2}

| Model | Accuracy | Out-of-scope recall | Model size (MB) | Mean CPU latency (ms) |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.5480 | 0.8548 | 25.28 | 7.41 |
| neural | 0.2750 | 0.7903 | 5.20 | 0.22 |
| teacher | 0.7139 | 0.7888 | 1060.84 | 107.03 |
| student | 0.6329 | 0.8556 | 516.40 | 59.89 |
| student_int8 | 0.6393 | 0.8164 | 393.14 | 15.45 |
