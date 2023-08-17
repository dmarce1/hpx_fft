#define       X              %rdi
#define       Y              %rsi
#define       I              %r9
#define       N              %rdx
#define       C              %ymm0
#define       S              %ymm1
#define       er             %ymm2
#define       ei             %ymm3
#define       tr             %ymm4
#define       ti             %ymm5

              .global        apply_twiddles_1d

              .text

apply_twiddles_1d:
              vbroadcastsd   %xmm0, C
              vbroadcastsd   %xmm1, S
              xor            I, I
loop:         vmovapd        (X, I, 8), er
              vmovapd        (Y, I, 8), ei
              vmulpd         S, ei, tr
              vmulpd         C, ei, ti
              vfmsub231pd    C, er, tr
              vfmadd231pd    S, er, ti
              vmovapd        tr, (X, I, 8)
              vmovapd        ti, (Y, I, 8)
              add            $4, I
              cmp            I, N
              jne            loop
              ret
