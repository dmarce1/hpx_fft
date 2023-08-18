#define       X              %r15
#define       Y              %r14
#define       N              %r13
#define       N2             %r12
#define       ky             %r11
#define       kz             %r10
#define       Wr             %r9
#define       Wi             %r8
#define       N0             %rsi
#define       N2N0           %rdi
#define       ER00           %ymm0
#define       ER01           %ymm1
#define       ER10           %ymm2
#define       ER11           %ymm3
#define       EI00           %ymm4
#define       EI01           %ymm5
#define       EI10           %ymm6
#define       EI11           %ymm7
#define       TR00           %ymm8
#define       TI00           %ymm9
#define       COS01          %ymm10
#define       COS10          %ymm11
#define       COS11          %ymm12
#define       SIN01          %ymm13
#define       SIN10          %ymm14
#define       SIN11          %ymm15
#define       STACK_SIZE     $16
#define       Wrptr          -8(%rbp)
#define       Wiptr          -16(%rbp)


              .global        fft_vector_2d

              .text

fft_vector_2d:push           %rbp
              mov            %rsp, %rbp
              sub            STACK_SIZE, %rsp
              push           %r15
              push           %r14
              push           %r13
              push           %r12
              push           %rbx
              mov            %rdi, X
              mov            %rsi, Y
              mov            %rdx, Wrptr
              mov            %rcx, Wiptr
              mov            %r8, N
              mov            %r9, N0
              mov            %r8, N2
              shr            N2
              call           next_level
              pop            %rbx
              pop            %r12
              pop            %r13
              pop            %r14
              pop            %r15
              mov            %rbp, %rsp
              pop            %rbp
              ret
next_level:   cmp            $4, N
              jg             rec
              mov            N0, %rdx
              lea            (X), %rax
              lea            (%rax, %rdx, 8), %rbx
              lea            (%rbx, %rdx, 8), %rcx
              vmovapd        (%rax), %ymm0
              vmovapd        (%rbx), %ymm1
              vmovapd        (%rcx), %ymm2
              vmovapd        (%rcx, %rdx, 8), %ymm3
              lea            (Y), %rax
              lea            (%rax, %rdx, 8), %rbx
              lea            (%rbx, %rdx, 8), %rcx
              vmovapd        (%rax), %ymm4
              vmovapd        (%rbx), %ymm5
              vmovapd        (%rcx), %ymm6
              vmovapd        (%rcx, %rdx, 8), %ymm7
              call           butterfly4
	          call           transpose
              call           butterfly4
              call           transpose
              vmovapd        %ymm4, (%rax)
              vmovapd        %ymm5, (%rbx)
              vmovapd        %ymm6, (%rcx)
              vmovapd        %ymm7, (%rcx, %rdx, 8)
              lea            (X), %rax
              lea            (%rax, %rdx, 8), %rbx
              lea            (%rbx, %rdx, 8), %rcx
              vmovapd        %ymm0, (%rax)
              vmovapd        %ymm1, (%rbx)
              vmovapd        %ymm2, (%rcx)
              vmovapd        %ymm3, (%rcx, %rdx, 8)
              jmp            done
rec:          mov            N, %rax
              imul           N0, %rax
              sub            N, %rax
              push           Y
              push           X
              lea            (X, N, 4), X
              lea            (Y, N, 4), Y
              push           Y
              push           X
              lea            (X, %rax, 4), X
              lea            (Y, %rax, 4), Y
              push           Y
              push           X
              lea            (X, N, 4), X
              lea            (Y, N, 4), Y
              shr            N
              shr            N2
              call           next_level
              pop            X
              pop            Y
              call           next_level
              pop            X
              pop            Y
              call           next_level
              pop            X
              pop            Y
              call           next_level
              shl            N2
              shl            N
              bsr            N, %rax
              mov            Wrptr, %rbx
              mov            Wiptr, %rcx
              mov            (%rbx, %rax, 8), Wr
              mov            (%rcx, %rax, 8), Wi
              mov            N2, %rax
              imul           N0, %rax
              mov            %rax, N2N0
              xor            ky, ky
kyloop:       vbroadcastsd   (Wr, ky, 8), COS10
              vbroadcastsd   (Wi, ky, 8), SIN10
              xor            kz, kz
kzloop:       vmovapd        (Wr, kz, 8), COS01
              vmovapd        (Wi, kz, 8), SIN01
              vmulpd         SIN10, SIN01, COS11
              vmulpd         COS10, SIN01, SIN11
              vfmsub231pd    COS10, COS01, COS11
              vfmadd231pd    SIN10, COS01, SIN11
              mov            ky, %rax
              imul           N0, %rax
              add            kz, %rax
              mov            %rax, %rbx
              mov            %rax, %rcx
              mov            %rax, %rdx
              add            N2, %rbx
              add            N2, %rdx
              add            N2N0, %rcx
              add            N2N0, %rdx
              vmovapd        (X, %rax, 8), ER00
              vmovapd        (Y, %rax, 8), EI00
              vmovapd        (X, %rbx, 8), ER01
              vmovapd        (Y, %rbx, 8), EI01
              vmovapd        (X, %rcx, 8), ER10
              vmovapd        (Y, %rcx, 8), EI10
              vmovapd        (X, %rdx, 8), ER11
              vmovapd        (Y, %rdx, 8), EI11
	          vmovapd        ER10, TR00
              vmovapd        EI10, TI00
              vmulpd         SIN10, TI00, ER10
              vmulpd         COS10, TI00, EI10
              vfmsub231pd    COS10, TR00, ER10
              vfmadd231pd    SIN10, TR00, EI10
              vmovapd        ER01, TR00
              vmovapd        EI01, TI00
              vmulpd         SIN01, TI00, ER01
              vmulpd         COS01, TI00, EI01
              vfmsub231pd    COS01, TR00, ER01
              vfmadd231pd    SIN01, TR00, EI01
              vmovapd        ER11, TR00
              vmovapd        EI11, TI00
              vmulpd         SIN11, TI00, ER11
              vmulpd         COS11, TI00, EI11
              vfmsub231pd    COS11, TR00, ER11
              vfmadd231pd    SIN11, TR00, EI11
              vmovapd        ER00, TR00
              vmovapd        EI00, TI00
              vaddpd         ER01, ER00, ER00
              vaddpd         EI01, EI00, EI00
              vsubpd         ER01, TR00, ER01
              vsubpd         EI01, TI00, EI01
              vmovapd        ER10, TR00
              vmovapd        EI10, TI00
              vaddpd         ER11, ER10, ER10
              vaddpd         EI11, EI10, EI10
              vsubpd         ER11, TR00, ER11
              vsubpd         EI11, TI00, EI11
              vmovapd        ER00, TR00
              vmovapd        EI00, TI00
              vaddpd         ER10, ER00, ER00
              vaddpd         EI10, EI00, EI00
              vsubpd         ER10, TR00, ER10
              vsubpd         EI10, TI00, EI10
              vmovapd        ER01, TR00
              vmovapd        EI01, TI00
              vaddpd         ER11, ER01, ER01
              vaddpd         EI11, EI01, EI01
              vsubpd         ER11, TR00, ER11
              vsubpd         EI11, TI00, EI11
              vmovapd        ER00, (X, %rax, 8)
              vmovapd        EI00, (Y, %rax, 8)
              vmovapd        ER01, (X, %rbx, 8)
              vmovapd        EI01, (Y, %rbx, 8)
              vmovapd        ER10, (X, %rcx, 8)
              vmovapd        EI10, (Y, %rcx, 8)
              vmovapd        ER11, (X, %rdx, 8)
              vmovapd        EI11, (Y, %rdx, 8)
              add            $4, kz
              cmp            kz, N2
              jne            kzloop
              inc            ky
              cmp            ky, N2
              jne            kyloop
done:         ret
transpose:    vunpcklpd      %ymm1, %ymm0, %ymm12
              vunpckhpd      %ymm1, %ymm0, %ymm13
              vunpcklpd      %ymm3, %ymm2, %ymm14
              vunpckhpd      %ymm3, %ymm2, %ymm15
              vperm2f128     $0x20, %ymm14, %ymm12, %ymm0
              vperm2f128     $0x20, %ymm15, %ymm13, %ymm1
              vperm2f128     $0x31, %ymm14, %ymm12, %ymm2
              vperm2f128     $0x31, %ymm15, %ymm13, %ymm3
              vunpcklpd      %ymm5, %ymm4, %ymm12
              vunpckhpd      %ymm5, %ymm4, %ymm13
              vunpcklpd      %ymm7, %ymm6, %ymm14
              vunpckhpd      %ymm7, %ymm6, %ymm15
              vperm2f128     $0x20, %ymm14, %ymm12, %ymm4
              vperm2f128     $0x20, %ymm15, %ymm13, %ymm5
              vperm2f128     $0x31, %ymm14, %ymm12, %ymm6
              vperm2f128     $0x31, %ymm15, %ymm13, %ymm7
              ret
butterfly4:   vaddpd         %ymm1, %ymm0, %ymm8
              vaddpd         %ymm3, %ymm2, %ymm10
              vaddpd         %ymm5, %ymm4, %ymm12
              vaddpd         %ymm7, %ymm6, %ymm14
              vsubpd         %ymm1, %ymm0, %ymm9
              vsubpd         %ymm3, %ymm2, %ymm11
              vsubpd         %ymm5, %ymm4, %ymm13
              vsubpd         %ymm7, %ymm6, %ymm15
              vaddpd         %ymm10, %ymm8, %ymm0
              vaddpd         %ymm15, %ymm9, %ymm1
              vaddpd         %ymm14, %ymm12, %ymm4
              vsubpd         %ymm11, %ymm13, %ymm5
              vsubpd         %ymm10, %ymm8, %ymm2
              vsubpd         %ymm15, %ymm9, %ymm3
              vsubpd         %ymm14, %ymm12, %ymm6
              vaddpd         %ymm11, %ymm13, %ymm7
              ret
