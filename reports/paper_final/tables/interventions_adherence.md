<!-- Target-specific adherence to one-control interventions (song-level mean over seeds; 95% bootstrap) -->
| model | condition | measure | intervened | N songs | samples | same measure, unmodified samples |
|---|---|---|---|---|---|---|
| midi_llm | bars_p4 | target section has the requested label+bars | 1.000 [1.000, 1.000] | 60 | 60 | 1.000 [1.000, 1.000] |
| midi_llm | bars_p4 | bar change in the requested direction | 1.000 [1.000, 1.000] | 60 | 60 |  |
| midi_llm | bars_p4 | non-target sections exact | 1.000 [1.000, 1.000] | 60 | 60 | 1.000 [1.000, 1.000] |
| midi_llm | bars_p4 | whole plan exact | 1.000 [1.000, 1.000] | 60 | 60 | 1.000 [1.000, 1.000] |
| midi_llm | key_p5 | declared key = requested | 1.000 [1.000, 1.000] | 60 | 60 |  |
| midi_llm | key_p5 | melody time in the new key's scale | 0.975 [0.941, 0.998] | 60 | 60 |  |
| midi_llm | key_p5 | melody time in the old key's scale | 0.934 [0.909, 0.953] | 60 | 60 |  |
| midi_llm | key_p5 | chord roots in the new key | 0.975 [0.943, 0.997] | 60 | 60 |  |
| midi_llm | key_p5 | pitch-class profile shifted by the requested interval | 0.537 [0.450, 0.625] | 60 | 60 |  |
| midi_llm | key_p5 | mean pitch shift (semitones) | -0.588 [-1.716, 0.536] | 60 | 60 |  |
| midi_llm | key_p5 | whole plan exact | 1.000 [1.000, 1.000] | 60 | 60 |  |
| midi_llm | label_bridge | target section has the requested label | 1.000 [1.000, 1.000] | 60 | 60 | 1.000 [1.000, 1.000] |
| midi_llm | label_bridge | target duration preserved | 1.000 [1.000, 1.000] | 60 | 60 | 1.000 [1.000, 1.000] |
| midi_llm | label_bridge | non-target sections exact | 1.000 [1.000, 1.000] | 60 | 60 | 1.000 [1.000, 1.000] |
| midi_llm | label_bridge | whole plan exact | 1.000 [1.000, 1.000] | 60 | 60 | 1.000 [1.000, 1.000] |
| midi_llm | lyrics_all | new lyrics sung (recall) | 0.851 [0.826, 0.872] | 59 | 59 |  |
| midi_llm | lyrics_all | chance: new lyrics 'sung' by unmodified samples | 0.106 [0.099, 0.112] | 59 | 59 |  |
| midi_llm | lyrics_all | old lyrics still sung | 0.108 [0.101, 0.115] | 59 | 59 |  |
| midi_llm | lyrics_all | whole plan exact | 0.983 [0.950, 1.000] | 60 | 60 | 1.000 [1.000, 1.000] |
| midi_llm | tempo_x1.25 | tempo header = requested | 1.000 [1.000, 1.000] | 60 | 60 |  |
| midi_llm | tempo_x1.25 | duration ratio (new/orig) | 0.800 [0.799, 0.800] | 60 | 60 |  |
| midi_llm | tempo_x1.25 | requested duration ratio | 0.800 [0.799, 0.800] | 60 | 60 |  |
| midi_llm | tempo_x1.25 | bar count = requested | 1.000 [1.000, 1.000] | 60 | 60 |  |
| midi_llm | tempo_x1.25 | whole plan exact | 1.000 [1.000, 1.000] | 60 | 60 |  |
| mupt | bars_m4 | target section has the requested label+bars | 0.830 [0.790, 0.868] | 224 | 448 | 0.825 [0.792, 0.856] |
| mupt | bars_m4 | bar change in the requested direction | 0.926 [0.899, 0.951] | 223 | 428 |  |
| mupt | bars_m4 | non-target sections exact | 0.738 [0.710, 0.765] | 224 | 448 | 0.789 [0.769, 0.808] |
| mupt | bars_m4 | whole plan exact | 0.272 [0.228, 0.317] | 224 | 448 | 0.343 [0.307, 0.379] |
| mupt | bars_p4 | target section has the requested label+bars | 0.812 [0.772, 0.853] | 224 | 448 | 0.825 [0.792, 0.856] |
| mupt | bars_p4 | bar change in the requested direction | 0.905 [0.870, 0.936] | 220 | 416 |  |
| mupt | bars_p4 | non-target sections exact | 0.787 [0.761, 0.812] | 224 | 448 | 0.789 [0.769, 0.808] |
| mupt | bars_p4 | whole plan exact | 0.321 [0.275, 0.371] | 224 | 448 | 0.343 [0.307, 0.379] |
| mupt | key_p5 | declared key = requested | 1.000 [1.000, 1.000] | 225 | 450 |  |
| mupt | key_p5 | melody time in the new key's scale | 0.967 [0.955, 0.977] | 225 | 450 |  |
| mupt | key_p5 | melody time in the old key's scale | 0.890 [0.875, 0.904] | 225 | 450 |  |
| mupt | key_p5 | chord roots in the new key | 0.922 [0.906, 0.938] | 225 | 450 |  |
| mupt | key_p5 | pitch-class profile shifted by the requested interval | 0.388 [0.358, 0.418] | 225 | 450 |  |
| mupt | key_p5 | mean pitch shift (semitones) | -0.088 [-0.490, 0.309] | 225 | 450 |  |
| mupt | key_p5 | whole plan exact | 0.322 [0.276, 0.369] | 225 | 450 |  |
| mupt | label_bridge | target section has the requested label | 0.801 [0.759, 0.842] | 224 | 448 | 0.859 [0.830, 0.887] |
| mupt | label_bridge | target duration preserved | 0.828 [0.788, 0.866] | 224 | 448 | 0.842 [0.810, 0.872] |
| mupt | label_bridge | non-target sections exact | 0.745 [0.716, 0.772] | 224 | 448 | 0.789 [0.769, 0.808] |
| mupt | label_bridge | whole plan exact | 0.306 [0.259, 0.353] | 224 | 448 | 0.343 [0.307, 0.379] |
| mupt | lyrics_all | new lyrics sung (recall) | 0.496 [0.480, 0.512] | 225 | 900 |  |
| mupt | lyrics_all | chance: new lyrics 'sung' by unmodified samples | 0.100 [0.096, 0.103] | 225 | 900 |  |
| mupt | lyrics_all | old lyrics still sung | 0.102 [0.099, 0.105] | 225 | 900 |  |
| mupt | lyrics_all | whole plan exact | 0.372 [0.332, 0.413] | 225 | 900 | 0.343 [0.307, 0.379] |
| mupt | tempo_x1.25 | tempo header = requested | 0.840 [0.793, 0.884] | 225 | 450 |  |
| mupt | tempo_x1.25 | duration ratio (new/orig) | 2.296 [1.660, 3.021] | 225 | 450 |  |
| mupt | tempo_x1.25 | requested duration ratio | 0.800 [0.799, 0.800] | 225 | 450 |  |
| mupt | tempo_x1.25 | bar count = requested | 0.342 [0.296, 0.389] | 225 | 450 |  |
| mupt | tempo_x1.25 | whole plan exact | 0.311 [0.267, 0.358] | 225 | 450 |  |
| qwen_e3b | bars_m2 | target section has the requested label+bars | 0.998 [0.993, 1.000] | 224 | 448 | 0.996 [0.991, 0.999] |
| qwen_e3b | bars_m2 | bar change in the requested direction | 1.000 [1.000, 1.000] | 224 | 448 |  |
| qwen_e3b | bars_m2 | non-target sections exact | 0.997 [0.994, 0.999] | 224 | 448 | 0.996 [0.992, 0.999] |
| qwen_e3b | bars_m2 | whole plan exact | 0.980 [0.967, 0.991] | 224 | 448 | 0.983 [0.974, 0.991] |
| qwen_e3b | bars_m4 | target section has the requested label+bars | 0.991 [0.982, 0.998] | 224 | 448 | 0.996 [0.991, 0.999] |
| qwen_e3b | bars_m4 | bar change in the requested direction | 0.993 [0.984, 1.000] | 224 | 448 |  |
| qwen_e3b | bars_m4 | non-target sections exact | 0.998 [0.994, 1.000] | 224 | 448 | 0.996 [0.992, 0.999] |
| qwen_e3b | bars_m4 | whole plan exact | 0.978 [0.964, 0.991] | 224 | 448 | 0.983 [0.974, 0.991] |
| qwen_e3b | bars_p2 | target section has the requested label+bars | 1.000 [1.000, 1.000] | 224 | 448 | 0.996 [0.991, 0.999] |
| qwen_e3b | bars_p2 | bar change in the requested direction | 1.000 [1.000, 1.000] | 224 | 448 |  |
| qwen_e3b | bars_p2 | non-target sections exact | 0.998 [0.997, 0.999] | 224 | 448 | 0.996 [0.992, 0.999] |
| qwen_e3b | bars_p2 | whole plan exact | 0.980 [0.967, 0.991] | 224 | 448 | 0.983 [0.974, 0.991] |
| qwen_e3b | bars_p4 | target section has the requested label+bars | 1.000 [1.000, 1.000] | 224 | 448 | 0.996 [0.991, 0.999] |
| qwen_e3b | bars_p4 | bar change in the requested direction | 1.000 [1.000, 1.000] | 224 | 448 |  |
| qwen_e3b | bars_p4 | non-target sections exact | 0.998 [0.996, 0.999] | 224 | 448 | 0.996 [0.992, 0.999] |
| qwen_e3b | bars_p4 | whole plan exact | 0.980 [0.967, 0.991] | 224 | 448 | 0.983 [0.974, 0.991] |
| qwen_e3b | key_m3 | declared key = requested | 1.000 [1.000, 1.000] | 225 | 450 |  |
| qwen_e3b | key_m3 | melody time in the new key's scale | 0.965 [0.955, 0.975] | 225 | 450 |  |
| qwen_e3b | key_m3 | melody time in the old key's scale | 0.643 [0.627, 0.659] | 225 | 450 |  |
| qwen_e3b | key_m3 | chord roots in the new key | 0.924 [0.910, 0.938] | 225 | 450 |  |
| qwen_e3b | key_m3 | pitch-class profile shifted by the requested interval | 0.364 [0.336, 0.392] | 225 | 450 |  |
| qwen_e3b | key_m3 | mean pitch shift (semitones) | 0.761 [0.340, 1.200] | 225 | 450 |  |
| qwen_e3b | key_m3 | whole plan exact | 0.982 [0.969, 0.993] | 225 | 450 |  |
| qwen_e3b | key_p2 | declared key = requested | 1.000 [1.000, 1.000] | 225 | 450 |  |
| qwen_e3b | key_p2 | melody time in the new key's scale | 0.971 [0.962, 0.979] | 225 | 450 |  |
| qwen_e3b | key_p2 | melody time in the old key's scale | 0.782 [0.769, 0.795] | 225 | 450 |  |
| qwen_e3b | key_p2 | chord roots in the new key | 0.923 [0.908, 0.937] | 225 | 450 |  |
| qwen_e3b | key_p2 | pitch-class profile shifted by the requested interval | 0.409 [0.380, 0.439] | 225 | 450 |  |
| qwen_e3b | key_p2 | mean pitch shift (semitones) | 0.547 [0.128, 0.983] | 225 | 450 |  |
| qwen_e3b | key_p2 | whole plan exact | 0.989 [0.978, 0.998] | 225 | 450 |  |
| qwen_e3b | key_p5 | declared key = requested | 1.000 [1.000, 1.000] | 225 | 450 |  |
| qwen_e3b | key_p5 | melody time in the new key's scale | 0.974 [0.965, 0.982] | 225 | 450 |  |
| qwen_e3b | key_p5 | melody time in the old key's scale | 0.907 [0.894, 0.920] | 225 | 450 |  |
| qwen_e3b | key_p5 | chord roots in the new key | 0.931 [0.916, 0.945] | 225 | 450 |  |
| qwen_e3b | key_p5 | pitch-class profile shifted by the requested interval | 0.409 [0.377, 0.441] | 225 | 450 |  |
| qwen_e3b | key_p5 | mean pitch shift (semitones) | 0.523 [0.121, 0.931] | 225 | 450 |  |
| qwen_e3b | key_p5 | whole plan exact | 0.982 [0.967, 0.993] | 225 | 450 |  |
| qwen_e3b | label_bridge | target section has the requested label | 0.998 [0.993, 1.000] | 224 | 448 | 0.998 [0.994, 1.000] |
| qwen_e3b | label_bridge | target duration preserved | 0.996 [0.989, 1.000] | 224 | 448 | 0.996 [0.991, 0.999] |
| qwen_e3b | label_bridge | non-target sections exact | 0.996 [0.993, 0.999] | 224 | 448 | 0.996 [0.992, 0.999] |
| qwen_e3b | label_bridge | whole plan exact | 0.971 [0.955, 0.984] | 224 | 448 | 0.983 [0.974, 0.991] |
| qwen_e3b | label_swap | target section has the requested label | 1.000 [1.000, 1.000] | 224 | 448 | 0.998 [0.994, 1.000] |
| qwen_e3b | label_swap | target duration preserved | 0.998 [0.993, 1.000] | 224 | 448 | 0.996 [0.991, 0.999] |
| qwen_e3b | label_swap | non-target sections exact | 0.996 [0.992, 0.999] | 224 | 448 | 0.996 [0.992, 0.999] |
| qwen_e3b | label_swap | whole plan exact | 0.980 [0.967, 0.991] | 224 | 448 | 0.983 [0.974, 0.991] |
| qwen_e3b | lyrics_all | new lyrics sung (recall) | 0.983 [0.979, 0.986] | 225 | 900 |  |
| qwen_e3b | lyrics_all | chance: new lyrics 'sung' by unmodified samples | 0.115 [0.112, 0.119] | 225 | 900 |  |
| qwen_e3b | lyrics_all | old lyrics still sung | 0.116 [0.113, 0.120] | 225 | 900 |  |
| qwen_e3b | lyrics_all | whole plan exact | 0.990 [0.983, 0.996] | 225 | 900 | 0.983 [0.974, 0.991] |
| qwen_e3b | lyrics_sec | new section lyrics sung in the target section | 0.982 [0.973, 0.989] | 224 | 448 |  |
| qwen_e3b | lyrics_sec | chance: same, unmodified samples | 0.087 [0.081, 0.092] | 224 | 448 |  |
| qwen_e3b | lyrics_sec | old section lyrics still sung there | 0.089 [0.082, 0.096] | 224 | 448 |  |
| qwen_e3b | lyrics_sec | whole plan exact | 0.987 [0.975, 0.996] | 224 | 448 | 0.983 [0.974, 0.991] |
| qwen_e3b | tempo_x0.8 | tempo header = requested | 1.000 [1.000, 1.000] | 225 | 450 |  |
| qwen_e3b | tempo_x0.8 | duration ratio (new/orig) | 1.253 [1.249, 1.258] | 225 | 450 |  |
| qwen_e3b | tempo_x0.8 | requested duration ratio | 1.250 [1.249, 1.250] | 225 | 450 |  |
| qwen_e3b | tempo_x0.8 | bar count = requested | 0.989 [0.978, 0.998] | 225 | 450 |  |
| qwen_e3b | tempo_x0.8 | whole plan exact | 0.984 [0.971, 0.996] | 225 | 450 |  |
| qwen_e3b | tempo_x1.25 | tempo header = requested | 1.000 [1.000, 1.000] | 225 | 450 |  |
| qwen_e3b | tempo_x1.25 | duration ratio (new/orig) | 0.801 [0.799, 0.805] | 225 | 450 |  |
| qwen_e3b | tempo_x1.25 | requested duration ratio | 0.800 [0.799, 0.800] | 225 | 450 |  |
| qwen_e3b | tempo_x1.25 | bar count = requested | 0.984 [0.971, 0.996] | 225 | 450 |  |
| qwen_e3b | tempo_x1.25 | whole plan exact | 0.973 [0.958, 0.987] | 225 | 450 |  |
