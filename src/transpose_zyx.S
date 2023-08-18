#define       X              %r8
#define       M              %r9
#define       Ix             %r10
#define       Iy             %r11
#define       IMID           %r12
#define       I1             %r13
#define       I2             %r14
#define       N1N2           %r15
#define       M0             $4
#define       STACK_SIZE     $16
#define       N1             -8(%rbp)
#define       N2             -16(%rbp)




              .global        transpose_re


              .text

transpose_re: enter          STACK_SIZE, $0
              push           %r12
              push           %r13
              push           %r14
              push           %r15
              push           %rbx
              mov            %rdi, X
              mov            %rsi, N1
              mov            %rdx, N2
              imul           N1, %rdx
              mov            %rdx, N1N2
              xor            IMID, IMID
mid_loop:     xor            Ix, Ix
              xor            Iy, Iy
              mov            N1, M
              call           next_level
              inc            IMID
              cmp            IMID, N2
              jne            mid_loop
              pop            %rbx
              pop            %r15
              pop            %r14
              pop            %r13
              pop            %r12
              leave
              ret
next_level:   cmp            Ix, Iy
              jl             done
              cmp            M0, M
              jg             next_levels
              imul           $2, N1N2, %rax
              imul           $3, N1N2, %rbx
              cmp            Ix, Iy
              je             on_diagonal
              mov            Ix, %rsi
              mov            Iy, %rdi
              imul           N2, %rsi
              imul           N2, %rdi
              add            IMID, %rsi
              add            IMID, %rdi
              imul           N1, %rsi
              imul           N1, %rdi
              add            Iy, %rsi
              add            Ix, %rdi
              lea            (X, %rsi, 8), %rsi
              lea            (X, %rdi, 8), %rdi
              vmovapd        (%rsi), %ymm0
              vmovapd        (%rsi, N1N2, 8), %ymm1
              vmovapd        (%rsi, %rax, 8), %ymm2
              vmovapd        (%rsi, %rbx, 8), %ymm3
              vmovapd        (%rdi), %ymm8
              vmovapd        (%rdi, N1N2, 8), %ymm9
              vmovapd        (%rdi, %rax, 8), %ymm10
              vmovapd        (%rdi, %rbx, 8), %ymm11
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
              vmovapd        %ymm1, (%rdi, N1N2, 8)
              vmovapd        %ymm2, (%rdi, %rax, 8)
              vmovapd        %ymm3, (%rdi, %rbx, 8)
              vmovapd        %ymm8, (%rsi)
              vmovapd        %ymm9, (%rsi, N1N2, 8)
              vmovapd        %ymm10, (%rsi, %rax, 8)
              vmovapd        %ymm11, (%rsi, %rbx, 8)
              jmp            done
on_diagonal:  mov            Ix, %rsi
              imul           N2, %rsi
              add            IMID, %rsi
              imul           N1, %rsi
              add            Ix, %rsi
              lea            (X, %rsi, 8), %rsi
              vmovapd        (%rsi), %ymm0
              vmovapd        (%rsi, N1N2, 8), %ymm1
              vmovapd        (%rsi, %rax, 8), %ymm2
              vmovapd        (%rsi, %rbx, 8), %ymm3
              vunpcklpd      %ymm1, %ymm0, %ymm4
              vunpckhpd      %ymm1, %ymm0, %ymm5
              vunpcklpd      %ymm3, %ymm2, %ymm6
              vunpckhpd      %ymm3, %ymm2, %ymm7
              vperm2f128     $0x20, %ymm6, %ymm4, %ymm0
              vperm2f128     $0x20, %ymm7, %ymm5, %ymm1
              vperm2f128     $0x31, %ymm6, %ymm4, %ymm2
              vperm2f128     $0x31, %ymm7, %ymm5, %ymm3
              vmovapd        %ymm0, (%rsi)
              vmovapd        %ymm1, (%rsi, N1N2, 8)
              vmovapd        %ymm2, (%rsi, %rax, 8)
              vmovapd        %ymm3, (%rsi, %rbx, 8)
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





