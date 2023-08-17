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
//#define       er2            %ymm2
//#define       er3            %ymm3
#define       ei0            %ymm4
#define       ei1            %ymm5
//#define       ei2            %ymm6
//#define       ei3            %ymm7
#define       cos1           %ymm8
#define       sin1           %ymm9
//#define       cos2           %ymm10
//#define       sin2           %ymm11
#define       tr             %ymm12
#define       ti             %ymm14
#define       two            %ymm15
#define       STACK_SIZE     $32
#define       twopioN        -8(%rbp)
#define       cos0           -16(%rbp)
#define       sin0           -24(%rbp)
#define       N0             -32(%rbp)


              .global        apply_fft_1d

              .text

              .align         32
TWO:          .double        2.0
              .double        2.0
              .double        2.0
              .double        2.0

apply_fft_1d: push           %rbp
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
              call           radix2
              pop            %rbx
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
              call           radix2
              pop            Y
              pop            X
              call           radix2
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
k2_loop:      mov            k2, %rax
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
              vmovq          cos0, %xmm0
              vmovq          sin0, %xmm1
              vbroadcastsd   %xmm0, cos1
              vbroadcastsd   %xmm1, sin1
              xor            I, I
I_loop:       mov            k2, %rdx
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
              jne            I_loop
              inc            k2
              cmp            k2, N2
              jne            k2_loop
              ret
