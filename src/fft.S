#define       X              %r15
#define       N              %r14
#define       Wr             %r13
#define       Wi             %r12
#define       k2             %r11
#define       N2             %r10
#define       NTW            %r9
#define       N28            %r8
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
#define       tr0            %ymm12
#define       ti0            %ymm13
#define       two            %ymm14

              .global        fft_1d

              .text
              .align         32
TWO:          .double        2.0
              .double        2.0
              .double        2.0
              .double        2.0

fft_1d:       push           %rbx
              push           %r15
              push           %r14
              push           %r13
              push           %r12
			  vmovapd        TWO, two
			  mov            %rdi, X
			  mov            %rsi, N
			  mov            %rdx, Wr
			  mov            %rcx, Wi
			  mov            $1, %rax
			  mov            %rax, NTW
			  bsr            N, %rax
			  test           $1, %rax
			  jz             do4
			  call           radix2
			  jmp            done
do4:          call           radix4
done:         pop            %r12
              pop            %r13
              pop            %r14
              pop            %r15
              pop            %rbx
              ret
radix2:       cmp            $2, N
              jl             radix2_end
              push           X
              mov            N, %rdi
              shl            $2, %rdi
              lea            (X, %rdi, 8), X
              shr            N
              shl            NTW
              call           radix4
              pop            X
              call           radix4
              shl            N
              shr            NTW
              mov            N, N2
              shr            N2
              mov            N2, N28
              shl            $3, N28
              xor            k2, k2
radix2_k2:    mov            k2, %rax
              imul           NTW, %rax
              vbroadcastsd   (Wr, %rax, 8), cos1
              vbroadcastsd   (Wi, %rax, 8), sin1
              mov            k2, %rdi
              shl            $3, %rdi
              lea            (X, %rdi, 8), %rax
              lea            (X, N28, 8), %rbx
              vmovapd        (%rax), er0
              vmovapd        32(%rax), ei0
              vmovapd        (%rbx), er1
              vmovapd        32(%rbx), ei1
              vmovapd        er0, tr0
              vmovapd        ei0, ti0
              vfmadd231pd    sin1, ei1, tr0
              vfnmadd231pd   sin1, er1, ti0
              vfnmadd132pd   cos1, tr0, er1
              vfnmadd132pd   cos1, ti0, ei1
              vfmsub132pd    two, er1, er0
              vfmsub132pd    two, ei1, ei0
              vmovapd        er0, (%rax)
              vmovapd        ei0, 32(%rax)
              vmovapd        er1, (%rbx)
              vmovapd        ei1, 32(%rbx)
              inc            k2
              cmp            k2, N2
              jne            radix2_k2
radix2_end:   ret
radix4:       cmp            $4, N
              jl             radix4_end
              push           X
              mov            N, %rdi
              shl            %rdi
              lea            (X, %rdi, 8), X
              push           X
              lea            (X, %rdi, 8), X
              push           X
              lea            (X, %rdi, 8), X
              shr            $2, N
              shl            $2, NTW
              call           radix4
              pop            X
              call           radix4
              pop            X
              call           radix4
              pop            X
              call           radix4
              shl            $2, N
              shr            $2, NTW
              mov            N, N2
              shr            $2, N2
              xor            k2, k2
              mov            N2, N28
              shl            $3, N28
radix4_k2:    mov            k2, %rax
              imul           NTW, %rax
              vbroadcastsd   (Wr, %rax, 8), cos1
              vbroadcastsd   (Wi, %rax, 8), sin1
              vmulpd         sin1, sin1, cos2
              vmulpd         cos1, sin1, sin2
              vfmsub231pd    cos1, cos1, cos2
              vfmadd231pd    sin1, cos1, sin2
              mov            k2, %rdi
              shl            $3, %rdi
              lea            (X, %rdi, 8), %rax
              lea            (%rax, N28, 8), %rbx
              lea            (%rbx, N28, 8), %rcx
              lea            (%rcx, N28, 8), %rdx
              vmovapd        (%rax), er0
              vmovapd        32(%rax), ei0
              vmovapd        (%rbx), er2
              vmovapd        32(%rbx), ei2
              vmovapd        (%rcx), er1
              vmovapd        32(%rcx), ei1
              vmovapd        (%rdx), er3
              vmovapd        32(%rdx), ei3
              vmovapd        er0, tr0
              vmovapd        ei0, ti0
              vfmadd231pd    sin2, ei2, tr0
              vfnmadd231pd   sin2, er2, ti0
              vfnmadd132pd   cos2, tr0, er2
              vfnmadd132pd   cos2, ti0, ei2
              vfmsub132pd    two, er2, er0
              vfmsub132pd    two, ei2, ei0
              vmovapd        er1, tr0
              vmovapd        ei1, ti0
              vfmadd231pd    sin2, ei3, tr0
              vfnmadd231pd   sin2, er3, ti0
              vfnmadd132pd   cos2, tr0, er3
              vfnmadd132pd   cos2, ti0, ei3
              vfmsub132pd    two, er3, er1
              vfmsub132pd    two, ei3, ei1
              vmovapd        er0, tr0
              vmovapd        ei0, ti0
              vfmadd231pd    sin1, ei1, tr0
              vfnmadd231pd   sin1, er1, ti0
              vfnmadd132pd   cos1, tr0, er1
              vfnmadd132pd   cos1, ti0, ei1
              vfmsub132pd    two, er1, er0
              vfmsub132pd    two, ei1, ei0
              vmovapd        er2, tr0
              vmovapd        ei2, ti0
              vfmadd231pd    cos1, ei3, tr0
              vfnmadd231pd   cos1, er3, ti0
              vfmadd132pd    sin1, tr0, er3
              vfmadd132pd    sin1, ti0, ei3
              vfmsub132pd    two, er3, er2
              vfmsub132pd    two, ei3, ei2
              vmovapd        er0, (%rax)
              vmovapd        ei0, 32(%rax)
              vmovapd        er3, (%rbx)
              vmovapd        ei3, 32(%rbx)
              vmovapd        er1, (%rcx)
              vmovapd        ei1, 32(%rcx)
              vmovapd        er2, (%rdx)
              vmovapd        ei2, 32(%rdx)
              inc            k2
              cmp            k2, N2
              jne            radix4_k2
radix4_end:   ret
