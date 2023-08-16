#define       X              %r8
#define       M              %r9
#define       Ix             %r10
#define       Iy             %r11
#define       Iz             %r12
#define       I1             %r13
#define       I2             %r14
#define       N              %r15
#define       M0             $4

              .global        transpose_zyx_asm


              .text

transpose_zyx_asm:
              push           %r12
              push           %r13
              push           %r14
              push           %r15
              push           %rbx
              mov            %rdi, X
              mov            %rsi, N
              mov            N, %rax
              imul           %rax, %rax
              imul           $2, %rax, %rbx
              imul           $3, %rax, %rcx
              mov            N, M
              xor            Iy, Iy
y_loop:       xor            Ix, Ix
              xor            Iz, Iz
              call           next_level
              inc            Iy
              cmp            Iy, N
              jne            y_loop
              pop            %rbx
              pop            %r15
              pop            %r14
              pop            %r13
              pop            %r12
              ret
next_level:   cmp            Ix, Iz
              jl             done
              cmp            M0, M
              jg             next_levels
              cmp            Ix, Iz
              je             on_diagonal
              mov            Ix, %rsi
              mov            Iz, %rdi
              imul           N, %rdi
              imul           N, %rsi
              add            Iy, %rsi
              add            Iy, %rdi
              imul           N, %rsi
              imul           N, %rdi
              add            Iz, %rsi
              add            Ix, %rdi
              lea            (X, %rsi, 8), %rsi
              lea            (X, %rdi, 8), %rdi
              vmovapd        (%rsi), %ymm0
              vmovapd        (%rsi, %rax, 8), %ymm1
              vmovapd        (%rsi, %rbx, 8), %ymm2
              vmovapd        (%rsi, %rcx, 8), %ymm3
              vmovapd        (%rdi), %ymm8
              vmovapd        (%rdi, %rax, 8), %ymm9
              vmovapd        (%rdi, %rbx, 8), %ymm10
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
              vmovapd        %ymm1, (%rdi, %rax, 8)
              vmovapd        %ymm2, (%rdi, %rbx, 8)
              vmovapd        %ymm3, (%rdi, %rcx, 8)
              vmovapd        %ymm8, (%rsi)
              vmovapd        %ymm9, (%rsi, %rax, 8)
              vmovapd        %ymm10, (%rsi, %rbx, 8)
              vmovapd        %ymm11, (%rsi, %rcx, 8)
              jmp            done
on_diagonal:  mov            Ix, %rsi
              imul           N, %rsi
              add            Iy, %rsi
              imul           N, %rsi
              add            Iz, %rsi
              lea            (X, %rsi, 8), %rsi
              vmovapd        (%rsi), %ymm0
              vmovapd        (%rsi, %rax, 8), %ymm1
              vmovapd        (%rsi, %rbx, 8), %ymm2
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
              vmovapd        %ymm1, (%rsi, %rax, 8)
              vmovapd        %ymm2, (%rsi, %rbx, 8)
              vmovapd        %ymm3, (%rsi, %rcx, 8)
              jmp            done
next_levels:  shr            M
              call           next_level
              add            M, Iz
              call           next_level
              sub            M, Iz
              add            M, Ix
              call           next_level
              add            M, Iz
              call           next_level
              sub            M, Iz
              sub            M, Ix
              shl            M
done:         ret





