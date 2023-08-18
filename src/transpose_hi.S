#define       X              %r8
#define       M              %r9
#define       Ix             %r10
#define       Iy             %r11
#define       I1             %r13
#define       I2             %r14
#define       N              %r15




              .global        transpose_hi


              .text

transpose_hi: push           %r12
              push           %r13
              push           %r14
              push           %r15
              push           %rbx
              mov            %rdi, X
              mov            %rsi, N
              xor            Ix, Ix
              xor            Iy, Iy
              mov            N, M
              call           next_level
              pop            %rbx
              pop            %r15
              pop            %r14
              pop            %r13
              pop            %r12
              ret
next_level:   cmp            Ix, Iy
              jl             done
              cmp            $1, M
              jg             next_levels
              cmp            Ix, Iy
              je             done
              mov            Ix, %rsi
              mov            Iy, %rdi
              imul           N, %rsi
              imul           N, %rdi
              add            Iy, %rsi
              add            Ix, %rdi
              imul           N, %rsi
              imul           N, %rdi
              mov            %rsi, %rcx
              add            N, %rcx
copy_loop1:   vmovapd        (X, %rsi, 8), %ymm0
              vmovapd        (X, %rdi, 8), %ymm1
              vmovapd        %ymm1, (X, %rsi, 8)
              vmovapd        %ymm0, (X, %rdi, 8)
              add            $4, %rsi
              add            $4, %rdi
              cmp            %rsi, %rcx
              jne            copy_loop1
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





