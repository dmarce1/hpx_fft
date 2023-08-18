#define       X              %r12
#define       N              %r13
#define       N1             %r14
#define       N2             %r15

              .global        scramble

              .text

scramble:     push           %r15
              push           %r14
              push           %r13
              push           %r12
              push           %rbx
              mov            %rdi, X
              mov            %rsi, N
              cmp            $64, N
              jle            small
              bsf            N, %rcx
              shr            %rcx
              mov            $1, N1
              shl            %rcx, N1
              bsf            N, %rax
              and            $1, %rax
              mov            $1, N2
              cmp            $0, %rax
              mov            $2, %rdx
              cmovne         %rdx, N2
              mov            X, %rdi
              mov            N1, %rsi
              mov            N1, %rdx
              imul           N2, %rdx
              call           scramble_hi
              mov            X, %rdi
              mov            N1, %rsi
              mov            N2, %rdx
              call           transpose_zyx
              mov            X, %rdi
              mov            N1, %rsi
              mov            N1, %rdx
              imul           N2, %rdx
              call           scramble_hi
              jmp            done
small:        xor            %rdx, %rdx
              xor            %rbx, %rbx
              mov            N, %r8
              dec            %r8
small_loop:   cmp            %rbx, %rdx
              jle            skip_swap
              mov            (X, %rbx, 8), %rcx
              xchg           %rcx, (X, %rdx, 8)
              mov            %rcx, (X, %rbx, 8)
skip_swap:    cmp            %rbx, %r8
              je             done
              mov            %rdx, %rax
              not            %rax
              and            %r8, %rax
              bsr            %rax, %rcx
              mov            $1, %rax
              shl            %rcx, %rax
              dec            %rax
              mov            %rdx, %rcx
              and            %rax, %rcx
              inc            %rax
              or             %rax, %rcx
              mov            %rcx, %rdx
              inc            %rbx
              jmp            small_loop
done:         pop            %rbx
              pop            %r12
              pop            %r13
              pop            %r14
              pop            %r15
              ret
