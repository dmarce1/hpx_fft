#define       X              %r8
#define       M              %r9
#define       Ix             %r10
#define       Iy             %r11
#define       I1             %r12
#define       N              %r13
#define       M0             $4




              .global        transpose_zy


              .text

transpose_zy: push           %r12
              push           %r13
              mov            %rdi, X
              mov            %rsi, N
              xor            Ix, Ix
              xor            Iy, Iy
              mov            N, M
              call           next_level
              pop            %r13
              pop            %r12
              ret
next_level:   cmp            Ix, Iy
              jl             done
              cmp            M0, M
              jg             next_levels
              imul           $2, N, %rax
              imul           $3, N, %rcx
              cmp            Ix, Iy
              je             on_diagonal
              mov            Ix, %rsi
              mov            Iy, %rdi
              imul           N, %rsi
              imul           N, %rdi
              add            Iy, %rsi
              add            Ix, %rdi
              lea            (X, %rsi, 8), %rsi
              lea            (X, %rdi, 8), %rdi
              vmovapd        (%rsi), %ymm0
              vmovapd        (%rsi, N, 8), %ymm1
              vmovapd        (%rsi, %rax, 8), %ymm2
              vmovapd        (%rsi, %rcx, 8), %ymm3
              vmovapd        (%rdi), %ymm8
              vmovapd        (%rdi, N, 8), %ymm9
              vmovapd        (%rdi, %rax, 8), %ymm10
              vmovapd        (%rdi, %rcx, 8), %ymm11
              vunpcklpd      %ymm1, %ymm0, %ymm4
              vunpckhpd      %ymm1, %ymm0, %ymm5
              vunpcklpd      %ymm3, %ymm2, %ymm6
              vunpckhpd      %ymm3, %ymm2, %ymm7
              vperm2f128     $0x20, %ymm6, %ymm4, %ymm0
              vperm2f128     $0x20, %ymm7, %ymm5, %ymm1
              vperm2f128     $0x31, %ymm6, %ymm4, %ymm2
              vperm2f128     $0x31, %ymm7, %ymm5, %ymm3
              vunpcklpd      %ymm9, %ymm8, %ymm4
              vunpckhpd      %ymm9, %ymm8, %ymm5
              vunpcklpd      %ymm11, %ymm10, %ymm6
              vunpckhpd      %ymm11, %ymm10, %ymm7
              vperm2f128     $0x20, %ymm6, %ymm4, %ymm8
              vperm2f128     $0x20, %ymm7, %ymm5, %ymm9
              vperm2f128     $0x31, %ymm6, %ymm4, %ymm10
              vperm2f128     $0x31, %ymm7, %ymm5, %ymm11
              vmovapd        %ymm0, (%rdi)
              vmovapd        %ymm1, (%rdi, N, 8)
              vmovapd        %ymm2, (%rdi, %rax, 8)
              vmovapd        %ymm3, (%rdi, %rcx, 8)
              vmovapd        %ymm8, (%rsi)
              vmovapd        %ymm9, (%rsi, N, 8)
              vmovapd        %ymm10, (%rsi, %rax, 8)
              vmovapd        %ymm11, (%rsi, %rcx, 8)
              jmp            done
on_diagonal:  mov            Ix, %rsi
              imul           N, %rsi
              add            Ix, %rsi
              lea            (X, %rsi, 8), %rsi
              vmovapd        (%rsi), %ymm0
              vmovapd        (%rsi, N, 8), %ymm1
              vmovapd        (%rsi, %rax, 8), %ymm2
              vmovapd        (%rsi, %rcx, 8), %ymm3
              vunpcklpd      %ymm1, %ymm0, %ymm4
              vunpckhpd      %ymm1, %ymm0, %ymm5
              vunpcklpd      %ymm3, %ymm2, %ymm6
              vunpckhpd      %ymm3, %ymm2, %ymm7
              vperm2f128     $0x20, %ymm6, %ymm4, %ymm0
              vperm2f128     $0x20, %ymm7, %ymm5, %ymm1
              vperm2f128     $0x31, %ymm6, %ymm4, %ymm2
              vperm2f128     $0x31, %ymm7, %ymm5, %ymm3
              vmovapd        %ymm0, (%rsi)
              vmovapd        %ymm1, (%rsi, N, 8)
              vmovapd        %ymm2, (%rsi, %rax, 8)
              vmovapd        %ymm3, (%rsi, %rcx, 8)
              jmp            done
next_levels:  shr            M
              call           next_level
              add            M, Iy
              call           next_level
              sub            M, Iy
              add            M, Ix
              call           next_level
              add            M, Iy
              call           next_level
              sub            M, Iy
              sub            M, Ix
              shl            M
done:         ret





