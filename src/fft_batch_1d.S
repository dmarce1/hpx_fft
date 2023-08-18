#define       X              %r15
#define       Y              %r14
#define       N              %r13
#define       M              %r12
#define       N2             %r11
#define       MN2            %r10
#define       k2             %r9
#define       I              %r8
#define       er0            %ymm0
#define       er1            %ymm1
#define       er2            %ymm2
#define       er3            %ymm3
#define       ei0            %ymm4
#define       ei1            %ymm5
#define       ei2            %ymm6
#define       ei3            %ymm7
#define       cos1           %ymm8
#define       sin1           %ymm9
#define       cos2           %ymm10
#define       sin2           %ymm11
#define       tr             %ymm12
#define       ti             %ymm14
#define       two            %ymm15
#define       STACK_SIZE     $32
#define       twopioN        -8(%rbp)
#define       cos0           -16(%rbp)
#define       sin0           -24(%rbp)
#define       N0             -32(%rbp)


              .global        fft_batch_1d

              .text

              .align         32
TWO:          .double        2.0
              .double        2.0
              .double        2.0
              .double        2.0

fft_batch_1d: push           %rbp
              mov            %rsp, %rbp
              sub            STACK_SIZE, %rsp
              push           %r15
              push           %r14
              push           %r13
              push           %r12
              push           %rbx
              mov            %rdi, X
              mov            %rsi, Y
              mov            %rdx, N
              mov            %rcx, M
              vmovapd        TWO, two
              bsr            N, %rax
              test           $1, %rax
              jnz            do2
              call           radix4
              jmp            fft_done
do2:          call           radix2
fft_done:     pop            %rbx
              pop            %r12
              pop            %r13
              pop            %r14
              pop            %r15
              mov            %rbp, %rsp
              pop            %rbp
              ret
radix2:       cmp            $2, N
              je             skip_2
              mov            N, %rax
              imul           M, %rax
              push           X
              push           Y
              lea            (X, %rax, 4), X
              lea            (Y, %rax, 4), Y
              shr            N
              call           radix4
              pop            Y
              pop            X
              call           radix4
              shl            N
skip_2:       fldpi
              fldpi
              faddp
              mov            N, N0
              fildl          N0
              fdivrp
              fchs
              fstpl          twopioN
              mov            N, N2
              shr            N2
              mov            N2, MN2
              imul           M, MN2
              xor            k2, k2
k2_loop_2:    mov            k2, %rax
              mov            %rax, cos0
              mov            %rax, sin0
              fildl          cos0
              fmull          twopioN
              fcos
              fstpl          cos0
              fildl          sin0
              fmull          twopioN
              fsin
              fstpl          sin0
              vbroadcastsd   cos0, cos1
              vbroadcastsd   sin0, sin1
              xor            I, I
I_loop_2:     mov            k2, %rdx
              imul           M, %rdx
              add            I, %rdx
              lea            (X, %rdx, 8), %rax
              lea            (Y, %rdx, 8), %rcx
              lea            (%rax, MN2, 8), %rbx
              lea            (%rcx, MN2, 8), %rdx
              vmovapd        (%rax), er0
              vmovapd        (%rbx), er1
              vmovapd        (%rcx), ei0
              vmovapd        (%rdx), ei1
              vmovapd        er0, tr
              vmovapd        ei0, ti
              vfmadd231pd    sin1, ei1, tr
              vfnmadd231pd   sin1, er1, ti
              vfnmadd132pd   cos1, tr, er1
              vfnmadd132pd   cos1, ti, ei1
              vfmsub132pd    two, er1, er0
              vfmsub132pd    two, ei1, ei0
              vmovapd        er0, (%rax)
              vmovapd        er1, (%rbx)
              vmovapd        ei0, (%rcx)
              vmovapd        ei1, (%rdx)
              add            $4, I
              cmp            I, M
              jne            I_loop_2
              inc            k2
              cmp            k2, N2
              jne            k2_loop_2
              ret
radix4:       cmp            $4, N
              je             skip_4
              mov            N, %rax
              imul           M, %rax
              push           X
              push           Y
              lea            (X, %rax, 2), X
              lea            (Y, %rax, 2), Y
              push           X
              push           Y
              lea            (X, %rax, 2), X
              lea            (Y, %rax, 2), Y
              push           X
              push           Y
              lea            (X, %rax, 2), X
              lea            (Y, %rax, 2), Y
              shr            $2, N
              call           radix4
              pop            Y
              pop            X
              call           radix4
              pop            Y
              pop            X
              call           radix4
              pop            Y
              pop            X
              call           radix4
              shl            $2, N
skip_4:       fldpi
              fldpi
              faddp
              mov            N, N0
              fildl          N0
              fdivrp
              fchs
              fstpl          twopioN
              mov            N, N2
              shr            $2, N2
              mov            N2, MN2
              imul           M, MN2
              shl            MN2
              xor            k2, k2
k2_loop_4:    mov            k2, %rax
              mov            %rax, cos0
              mov            %rax, sin0
              fildl          cos0
              fmull          twopioN
              fcos
              fstpl          cos0
              fildl          sin0
              fmull          twopioN
              fsin
              fstpl          sin0
              vbroadcastsd   cos0, cos1
              vbroadcastsd   sin0, sin1
              vmulpd         sin1, sin1, cos2
              vmulpd         cos1, sin1, sin2
              vfmsub231pd    cos1, cos1, cos2
              vfmadd231pd    sin1, cos1, sin2
              xor            I, I
I_loop_4:     mov            k2, %rdx
              imul           M, %rdx
              add            I, %rdx
              lea            (X, %rdx, 8), %rax
              lea            (Y, %rdx, 8), %rcx
              lea            (%rax, MN2, 8), %rbx
              lea            (%rcx, MN2, 8), %rdx
              vmovapd        (%rax), er0
              vmovapd        (%rbx), er1
              vmovapd        (%rcx), ei0
              vmovapd        (%rdx), ei1
              vmovapd        (%rax, MN2, 4), er2
              vmovapd        (%rbx, MN2, 4), er3
              vmovapd        (%rcx, MN2, 4), ei2
              vmovapd        (%rdx, MN2, 4), ei3
              vmovapd        er0, tr
              vmovapd        ei0, ti
              vfmadd231pd    sin2, ei2, tr
              vfnmadd231pd   sin2, er2, ti
              vfnmadd132pd   cos2, tr, er2
              vfnmadd132pd   cos2, ti, ei2
              vfmsub132pd    two, er2, er0
              vfmsub132pd    two, ei2, ei0
              vmovapd        er1, tr
              vmovapd        ei1, ti
              vfmadd231pd    sin2, ei3, tr
              vfnmadd231pd   sin2, er3, ti
              vfnmadd132pd   cos2, tr, er3
              vfnmadd132pd   cos2, ti, ei3
              vfmsub132pd    two, er3, er1
              vfmsub132pd    two, ei3, ei1
              vmovapd        er0, tr
              vmovapd        ei0, ti
              vfmadd231pd    sin1, ei1, tr
              vfnmadd231pd   sin1, er1, ti
              vfnmadd132pd   cos1, tr, er1
              vfnmadd132pd   cos1, ti, ei1
              vfmsub132pd    two, er1, er0
              vfmsub132pd    two, ei1, ei0
              vmovapd        er2, tr
              vmovapd        ei2, ti
              vfmadd231pd    cos1, ei3, tr
              vfnmadd231pd   cos1, er3, ti
              vfmadd132pd    sin1, tr, er3
              vfmadd132pd    sin1, ti, ei3
              vfmsub132pd    two, er3, er2
              vfmsub132pd    two, ei3, ei2
              vmovapd        er0, (%rax)
              vmovapd        er1, (%rbx)
              vmovapd        ei0, (%rcx)
              vmovapd        ei1, (%rdx)
              vmovapd        er3, (%rax, MN2, 4)
              vmovapd        er2, (%rbx, MN2, 4)
              vmovapd        ei3, (%rcx, MN2, 4)
              vmovapd        ei2, (%rdx, MN2, 4)
              add            $4, I
              cmp            I, M
              jne            I_loop_4
              inc            k2
              cmp            k2, N2
              jne            k2_loop_4
              ret
