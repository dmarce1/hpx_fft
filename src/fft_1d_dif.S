#define       X              %r15
#define       Y              %r14
#define       N              %r13
#define       Wr             %r12
#define       Wi             %r11
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
#define       tr0            %ymm8
#define       tr1            %ymm9
#define       tr2            %ymm10
#define       tr3            %ymm11
#define       ti0            %ymm12
#define       ti1            %ymm13
#define       ti2            %ymm14
#define       ti3            %ymm15
#define       STACK_SIZE     $216
#define       N2             -8(%rbp)
#define       M              -16(%rbp)
#define       NTW            -24(%rbp)
#define       cos1           -56(%rbp)
#define       sin1           -88(%rbp)
#define       cos2           -120(%rbp)
#define       sin2           -152(%rbp)
#define       cos3           -184(%rbp)
#define       sin3           -216(%rbp)


              .global        fft_1d_dif

              .text

              .align         32
TWO:          .double        2.0
              .double        2.0
              .double        2.0
              .double        2.0

fft_1d_dif:   push           %rbp
              mov            %rsp, %rbp
              sub            STACK_SIZE, %rsp
              push           %r15
              push           %r14
              push           %r13
              push           %r12
              push           %rbx
              mov            %rdi, X
              mov            %rsi, Y
              mov            %rdx, Wr
              mov            %rcx, Wi
              mov            %r8, N
              mov            %r9, M
              mov            $1, %rax
              mov            %rax, NTW
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
radix2:       mov            N, %rax
              shr            %rax
              mov            %rax, N2
              mov            %rax, MN2
              imul           M, MN2
              xor            k2, k2
k2_loop_2:    mov            k2, %rax
              imul           NTW, %rax
              vbroadcastsd   (Wr, %rax, 8), %ymm0
              vbroadcastsd   (Wi, %rax, 8), %ymm1
              vmovupd        %ymm0, cos1
              vmovupd        %ymm1, sin1
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
              vmovapd        er0, tr0
              vmovapd        ei0, ti0
              vaddpd         er1, tr0, er0
              vaddpd         ei1, ti0, ei0
              vsubpd         er1, tr0, er1
              vsubpd         ei1, ti0, ei1
              vmovapd        er1, tr0
              vmovapd        ei1, ti0
              vmulpd         sin1, ti0, er1
              vmulpd         cos1, ti0, ei1
              vfmsub231pd    cos1, tr0, er1
              vfmadd231pd    sin1, tr0, ei1
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
              cmp            $2, N
              je             skip_2
              mov            N, %rax
              imul           M, %rax
              push           X
              push           Y
              lea            (X, %rax, 4), X
              lea            (Y, %rax, 4), Y
              shr            N
              mov            NTW, %rax
              shl            %rax
              mov            %rax, NTW
              call           radix4
              pop            Y
              pop            X
              call           radix4
              mov            NTW, %rax
              shr            %rax
              mov            %rax, NTW
              shl            N
skip_2:       ret
radix4:       mov            N, %rax
              shr            $2, %rax
              mov            %rax, N2
              mov            %rax, MN2
              imul           M, MN2
              shl            MN2
              xor            k2, k2
k2_loop_4:    mov            k2, %rax
              imul           NTW, %rax
              vbroadcastsd   (Wr, %rax, 8), %ymm0
              vbroadcastsd   (Wi, %rax, 8), %ymm1
              vmulpd         %ymm1, %ymm1, %ymm2
              vmulpd         %ymm0, %ymm1, %ymm3
              vfmsub231pd    %ymm0, %ymm0, %ymm2
              vfmadd231pd    %ymm1, %ymm0, %ymm3
              vmulpd         %ymm1, %ymm3, %ymm4
              vmulpd         %ymm0, %ymm3, %ymm5
              vfmsub231pd    %ymm0, %ymm2, %ymm4
              vfmadd231pd    %ymm1, %ymm2, %ymm5
              vmovupd        %ymm0, cos1
              vmovupd        %ymm1, sin1
              vmovupd        %ymm2, cos2
              vmovupd        %ymm3, sin2
              vmovupd        %ymm4, cos3
              vmovupd        %ymm5, sin3
              xor            I, I
I_loop_4:     mov            k2, %rdx
              imul           M, %rdx
              add            I, %rdx
              lea            (X, %rdx, 8), %rax
              lea            (Y, %rdx, 8), %rcx
              lea            (%rax, MN2, 8), %rbx
              lea            (%rcx, MN2, 8), %rdx
              vmovapd        (%rax), er0
              vmovapd        (%rbx), er2
              vmovapd        (%rcx), ei0
              vmovapd        (%rdx), ei2
              vmovapd        (%rax, MN2, 4), er1
              vmovapd        (%rbx, MN2, 4), er3
              vmovapd        (%rcx, MN2, 4), ei1
              vmovapd        (%rdx, MN2, 4), ei3
              vaddpd         er2, er0, tr0
              vaddpd         ei2, ei0, ti0
              vsubpd         er2, er0, tr2
              vsubpd         ei2, ei0, ti2
              vaddpd         er3, er1, tr1
              vaddpd         ei3, ei1, ti1
              vsubpd         er3, er1, tr3
              vsubpd         ei3, ei1, ti3
              vaddpd         tr1, tr0, er0
              vaddpd         ti1, ti0, ei0
              vaddpd         ti3, tr2, er1
              vsubpd         tr3, ti2, ei1
              vsubpd         tr1, tr0, er2
              vsubpd         ti1, ti0, ei2
              vsubpd         ti3, tr2, er3
              vaddpd         tr3, ti2, ei3
              vmovapd        er1, tr0
              vmovapd        ei1, ti0
              vmulpd         sin1, ti0, er1
              vmulpd         cos1, ti0, ei1
              vfmsub231pd    cos1, tr0, er1
              vfmadd231pd    sin1, tr0, ei1
              vmovapd        er2, tr0
              vmovapd        ei2, ti0
              vmulpd         sin2, ti0, er2
              vmulpd         cos2, ti0, ei2
              vfmsub231pd    cos2, tr0, er2
              vfmadd231pd    sin2, tr0, ei2
              vmovapd        er3, tr0
              vmovapd        ei3, ti0
              vmulpd         sin3, ti0, er3
              vmulpd         cos3, ti0, ei3
              vfmsub231pd    cos3, tr0, er3
              vfmadd231pd    sin3, tr0, ei3
              vmovapd        er0, (%rax)
              vmovapd        er1, (%rbx)
              vmovapd        ei0, (%rcx)
              vmovapd        ei1, (%rdx)
              vmovapd        er2, (%rax, MN2, 4)
              vmovapd        er3, (%rbx, MN2, 4)
              vmovapd        ei2, (%rcx, MN2, 4)
              vmovapd        ei3, (%rdx, MN2, 4)
              add            $4, I
              cmp            I, M
              jne            I_loop_4
              inc            k2
              cmp            k2, N2
              jne            k2_loop_4
              cmp            $4, N
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
              mov            NTW, %rax
              shl            $2, %rax
              mov            %rax, NTW
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
              mov            NTW, %rax
              shr            $2, %rax
              mov            %rax, NTW
              shl            $2, N
skip_4:       ret
