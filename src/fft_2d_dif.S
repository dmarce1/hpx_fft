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
#define       CS01          %ymm10
#define       CS10          %ymm11
#define       CS11          %ymm12
#define       SN01          %ymm13
#define       SN10          %ymm14
#define       SN11          %ymm15
#define       STACK_SIZE     $16
#define       Wrptr          -8(%rbp)
#define       Wiptr          -16(%rbp)


              .global        fft_2d_dif

              .text

fft_2d_dif:   push           %rbp
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
              jg             interior
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
interior:     bsr            N, %rax
              mov            Wrptr, %rbx
              mov            Wiptr, %rcx
              mov            (%rbx, %rax, 8), Wr
              mov            (%rcx, %rax, 8), Wi
              mov            N2, %rax
              imul           N0, %rax
              mov            %rax, N2N0
              xor            ky, ky
kyloop:       vbroadcastsd   (Wr, ky, 8), CS10
              vbroadcastsd   (Wi, ky, 8), SN10
              xor            kz, kz
kzloop:       vmovapd        (Wr, kz, 8), CS01
              vmovapd        (Wi, kz, 8), SN01
              vmulpd         SN01, SN10, CS11
              vmulpd         CS01, SN10, SN11
              vfmsub231pd    CS01, CS10, CS11
              vfmadd231pd    SN01, CS10, SN11
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
              vmovapd        ER00, TR00
              vmovapd        EI00, TI00
              vaddpd         ER01, TR00, ER00
              vaddpd         EI01, TI00, EI00
              vsubpd         ER01, TR00, ER01
              vsubpd         EI01, TI00, EI01
              vmovapd        ER10, TR00
              vmovapd        EI10, TI00
              vaddpd         ER11, TR00, ER10
              vaddpd         EI11, TI00, EI10
              vsubpd         ER11, TR00, ER11
              vsubpd         EI11, TI00, EI11
	          vmovapd        ER00, TR00
              vmovapd        EI00, TI00
              vaddpd         ER10, TR00, ER00
              vaddpd         EI10, TI00, EI00
              vsubpd         ER10, TR00, ER10
              vsubpd         EI10, TI00, EI10
              vmovapd        ER01, TR00
              vmovapd        EI01, TI00
              vaddpd         ER11, TR00, ER01
              vaddpd         EI11, TI00, EI01
              vsubpd         ER11, TR00, ER11
              vsubpd         EI11, TI00, EI11
              vmovapd        ER01, TR00
              vmovapd        EI01, TI00
              vmulpd         SN01, TI00, ER01
              vmulpd         CS01, TI00, EI01
              vfmsub231pd    CS01, TR00, ER01
              vfmadd231pd    SN01, TR00, EI01
              vmovapd        ER10, TR00
              vmovapd        EI10, TI00
              vmulpd         SN10, TI00, ER10
              vmulpd         CS10, TI00, EI10
              vfmsub231pd    CS10, TR00, ER10
              vfmadd231pd    SN10, TR00, EI10
              vmovapd        ER11, TR00
              vmovapd        EI11, TI00
              vmulpd         SN11, TI00, ER11
              vmulpd         CS11, TI00, EI11
              vfmsub231pd    CS11, TR00, ER11
              vfmadd231pd    SN11, TR00, EI11
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
              mov            N, %rax
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
butterfly4:   vaddpd         %ymm2, %ymm0, %ymm8
              vaddpd         %ymm3, %ymm1, %ymm9
              vaddpd         %ymm6, %ymm4, %ymm12
              vaddpd         %ymm7, %ymm5, %ymm13
              vsubpd         %ymm2, %ymm0, %ymm10
              vsubpd         %ymm3, %ymm1, %ymm11
              vsubpd         %ymm6, %ymm4, %ymm14
              vsubpd         %ymm7, %ymm5, %ymm15
              vaddpd         %ymm9, %ymm8, %ymm0
              vaddpd         %ymm15, %ymm10, %ymm2
              vaddpd         %ymm13, %ymm12, %ymm4
              vsubpd         %ymm11, %ymm14, %ymm6
              vsubpd         %ymm9, %ymm8, %ymm1
              vsubpd         %ymm15, %ymm10, %ymm3
              vsubpd         %ymm13, %ymm12, %ymm5
              vaddpd         %ymm11, %ymm14, %ymm7
              ret
