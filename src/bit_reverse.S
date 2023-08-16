
              .global        bit_reverse


              .text

bit_reverse:  push           %rbp
              mov            %rsp, %rbp
              sub            $8, %rsp
              push           %rbx
              xor            %rax, %rax
              xor            %rbx, %rbx
              xor            %rcx, %rcx
              xor            %rdx, %rdx
              mov            %rdi, -8(%rbp)
              mov            -8(%rbp), %al
              mov            -7(%rbp), %bl
              mov            -6(%rbp), %cl
              mov            -5(%rbp), %dl
              lea            (table), %rdi
              mov            (%rdi, %rax), %al
              mov            (%rdi, %rbx), %bl
              mov            (%rdi, %rcx), %cl
              mov            (%rdi, %rdx), %dl
              mov            %al, -5(%rbp)
              mov            %bl, -6(%rbp)
              mov            %cl, -7(%rbp)
              mov            %dl, -8(%rbp)
              mov            -8(%rbp), %rax
              mov            $32, %rcx
              sub            %rsi, %rcx
              shr            %rcx, %rax
              pop            %rbx
              mov            %rbp, %rsp
              pop            %rbp
              ret
table:        .byte          0
              .byte          128
              .byte          64
              .byte          192
              .byte          32
              .byte          160
              .byte          96
              .byte          224
              .byte          16
              .byte          144
              .byte          80
              .byte          208
              .byte          48
              .byte          176
              .byte          112
              .byte          240
              .byte          8
              .byte          136
              .byte          72
              .byte          200
              .byte          40
              .byte          168
              .byte          104
              .byte          232
              .byte          24
              .byte          152
              .byte          88
              .byte          216
              .byte          56
              .byte          184
              .byte          120
              .byte          248
              .byte          4
              .byte          132
              .byte          68
              .byte          196
              .byte          36
              .byte          164
              .byte          100
              .byte          228
              .byte          20
              .byte          148
              .byte          84
              .byte          212
              .byte          52
              .byte          180
              .byte          116
              .byte          244
              .byte          12
              .byte          140
              .byte          76
              .byte          204
              .byte          44
              .byte          172
              .byte          108
              .byte          236
              .byte          28
              .byte          156
              .byte          92
              .byte          220
              .byte          60
              .byte          188
              .byte          124
              .byte          252
              .byte          2
              .byte          130
              .byte          66
              .byte          194
              .byte          34
              .byte          162
              .byte          98
              .byte          226
              .byte          18
              .byte          146
              .byte          82
              .byte          210
              .byte          50
              .byte          178
              .byte          114
              .byte          242
              .byte          10
              .byte          138
              .byte          74
              .byte          202
              .byte          42
              .byte          170
              .byte          106
              .byte          234
              .byte          26
              .byte          154
              .byte          90
              .byte          218
              .byte          58
              .byte          186
              .byte          122
              .byte          250
              .byte          6
              .byte          134
              .byte          70
              .byte          198
              .byte          38
              .byte          166
              .byte          102
              .byte          230
              .byte          22
              .byte          150
              .byte          86
              .byte          214
              .byte          54
              .byte          182
              .byte          118
              .byte          246
              .byte          14
              .byte          142
              .byte          78
              .byte          206
              .byte          46
              .byte          174
              .byte          110
              .byte          238
              .byte          30
              .byte          158
              .byte          94
              .byte          222
              .byte          62
              .byte          190
              .byte          126
              .byte          254
              .byte          1
              .byte          129
              .byte          65
              .byte          193
              .byte          33
              .byte          161
              .byte          97
              .byte          225
              .byte          17
              .byte          145
              .byte          81
              .byte          209
              .byte          49
              .byte          177
              .byte          113
              .byte          241
              .byte          9
              .byte          137
              .byte          73
              .byte          201
              .byte          41
              .byte          169
              .byte          105
              .byte          233
              .byte          25
              .byte          153
              .byte          89
              .byte          217
              .byte          57
              .byte          185
              .byte          121
              .byte          249
              .byte          5
              .byte          133
              .byte          69
              .byte          197
              .byte          37
              .byte          165
              .byte          101
              .byte          229
              .byte          21
              .byte          149
              .byte          85
              .byte          213
              .byte          53
              .byte          181
              .byte          117
              .byte          245
              .byte          13
              .byte          141
              .byte          77
              .byte          205
              .byte          45
              .byte          173
              .byte          109
              .byte          237
              .byte          29
              .byte          157
              .byte          93
              .byte          221
              .byte          61
              .byte          189
              .byte          125
              .byte          253
              .byte          3
              .byte          131
              .byte          67
              .byte          195
              .byte          35
              .byte          163
              .byte          99
              .byte          227
              .byte          19
              .byte          147
              .byte          83
              .byte          211
              .byte          51
              .byte          179
              .byte          115
              .byte          243
              .byte          11
              .byte          139
              .byte          75
              .byte          203
              .byte          43
              .byte          171
              .byte          107
              .byte          235
              .byte          27
              .byte          155
              .byte          91
              .byte          219
              .byte          59
              .byte          187
              .byte          123
              .byte          251
              .byte          7
              .byte          135
              .byte          71
              .byte          199
              .byte          39
              .byte          167
              .byte          103
              .byte          231
              .byte          23
              .byte          151
              .byte          87
              .byte          215
              .byte          55
              .byte          183
              .byte          119
              .byte          247
              .byte          15
              .byte          143
              .byte          79
              .byte          207
              .byte          47
              .byte          175
              .byte          111
              .byte          239
              .byte          31
              .byte          159
              .byte          95
              .byte          223
              .byte          63
              .byte          191
              .byte          127
              .byte          255
