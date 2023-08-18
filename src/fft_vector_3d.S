#define       X              %r15
#define       Y              %r14
#define       N              %r13
#define       M              %r12
#define       N2             %r11
#define       kx             %r10
#define       ky             %r9
#define       kz             %r8
#define       TR000          %ymm0
#define       ER001          %ymm1
#define       ER010          %ymm2
#define       ER011          %ymm3
#define       ER100          %ymm4
#define       ER101          %ymm5
#define       ER110          %ymm6
#define       ER111          %ymm7
#define       TI000          %ymm8
#define       EI001          %ymm9
#define       EI010          %ymm10
#define       EI011          %ymm11
#define       EI100          %ymm12
#define       EI101          %ymm13
#define       EI110          %ymm14
#define       EI111          %ymm15
#define       STACK_SIZE     $624
#define       ER000          -32(%rbp)
#define       EI000          -64(%rbp)
#define       COS001         -96(%rbp)
#define       COS010         -128(%rbp)
#define       COS011         -160(%rbp)
#define       COS100         -192(%rbp)
#define       COS101         -224(%rbp)
#define       COS110         -256(%rbp)
#define       COS111         -288(%rbp)
#define       SIN001         -320(%rbp)
#define       SIN010         -352(%rbp)
#define       SIN011         -384(%rbp)
#define       SIN100         -416(%rbp)
#define       SIN101         -448(%rbp)
#define       SIN110         -480(%rbp)
#define       SIN111         -512(%rbp)
#define       Wrptr          -520(%rbp)
#define       Wiptr          -528(%rbp)
#define       N0             -536(%rbp)
#define       xlb            -544(%rbp)
#define       ylb            -552(%rbp)
#define       zlb            -560(%rbp)
#define       xub            -568(%rbp)
#define       yub            -576(%rbp)
#define       zub            -584(%rbp)
#define       N2N0           -592(%rbp)
#define       N2N0N0         -600(%rbp)
#define       N0N0           -608(%rbp)
#define       Wr             -616(%rbp)
#define       Wi             -624(%rbp)


              .global        fft_vector_3d

              .text

NONE:         .double        -1.0
              .double        -1.0
              .double        -1.0
              .double        -1.0

fft_vector_3d:push           %rbp
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
              mov            %r8, N0
              mov            %r8, N2
              imul           %r8, %r8
              mov            %r8, N0N0
              shr            N2
              xor            %rax, %rax
              mov            %rax, xlb
              mov            %rax, ylb
              mov            %rax, zlb
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
              mov            xlb, %rax
              mov            ylb, %rbx
              mov            zlb, %rcx
              add            $4, %rax
              add            $4, %rbx
              add            $4, %rcx
              mov            %rax, xub
              mov            %rbx, yub
              mov            %rcx, zub
              mov            xlb, kx
loopy:        mov            kx, %rsi
              imul           N0, %rsi
              add            ylb, %rsi
              imul           N0, %rsi
              add            zlb, %rsi
              mov            N0, %rdx
              lea            (X, %rsi, 8), %rax
              lea            (%rax, %rdx, 8), %rbx
              lea            (%rbx, %rdx, 8), %rcx
              vmovapd        (%rax), %ymm0
              vmovapd        (%rbx), %ymm1
              vmovapd        (%rcx), %ymm2
              vmovapd        (%rcx, %rdx, 8), %ymm3
              lea            (Y, %rsi, 8), %rax
              lea            (%rax, %rdx, 8), %rbx
              lea            (%rbx, %rdx, 8), %rcx
              vmovapd        (%rax), %ymm4
              vmovapd        (%rbx), %ymm5
              vmovapd        (%rcx), %ymm6
              vmovapd        (%rcx, %rdx, 8), %ymm7
              call           butterfly4
              vunpcklpd      %ymm1, %ymm0, %ymm12
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
              call           butterfly4
              vunpcklpd      %ymm1, %ymm0, %ymm12
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
              vmovapd        %ymm4, (%rax)
              vmovapd        %ymm5, (%rbx)
              vmovapd        %ymm6, (%rcx)
              vmovapd        %ymm7, (%rcx, %rdx, 8)
              lea            (X, %rsi, 8), %rax
              lea            (%rax, %rdx, 8), %rbx
              lea            (%rbx, %rdx, 8), %rcx
              vmovapd        %ymm0, (%rax)
              vmovapd        %ymm1, (%rbx)
              vmovapd        %ymm2, (%rcx)
              vmovapd        %ymm3, (%rcx, %rdx, 8)
              inc            kx
              cmp            kx, xub
              jne            loopy
              mov            ylb, ky
loopx:        mov            xlb, %rsi
              imul           N0, %rsi
              add            ky, %rsi
              imul           N0, %rsi
              add            zlb, %rsi
              mov            N0N0, %rdx
              lea            (X, %rsi, 8), %rax
              lea            (%rax, %rdx, 8), %rbx
              lea            (%rbx, %rdx, 8), %rcx
              vmovapd        (%rax), %ymm0
              vmovapd        (%rbx), %ymm1
              vmovapd        (%rcx), %ymm2
              vmovapd        (%rcx, %rdx, 8), %ymm3
              lea            (Y, %rsi, 8), %rax
              lea            (%rax, %rdx, 8), %rbx
              lea            (%rbx, %rdx, 8), %rcx
              vmovapd        (%rax), %ymm4
              vmovapd        (%rbx), %ymm5
              vmovapd        (%rcx), %ymm6
              vmovapd        (%rcx, %rdx, 8), %ymm7
              call           butterfly4
              vmovapd        %ymm4, (%rax)
              vmovapd        %ymm5, (%rbx)
              vmovapd        %ymm6, (%rcx)
              vmovapd        %ymm7, (%rcx, %rdx, 8)
              lea            (X, %rsi, 8), %rax
              lea            (%rax, %rdx, 8), %rbx
              lea            (%rbx, %rdx, 8), %rcx
              vmovapd        %ymm0, (%rax)
              vmovapd        %ymm1, (%rbx)
              vmovapd        %ymm2, (%rcx)
              vmovapd        %ymm3, (%rcx, %rdx, 8)
              inc            ky
              cmp            ky, yub
              jne            loopx
              jmp            done
rec:          xor            %rax, %rax
              push           xlb
              add            %rax, xlb
              push           %rax
recxloop:     xor            %rbx, %rbx
              push           ylb
              add            %rbx, ylb
              push           %rbx
recyloop:     xor            %rcx, %rcx
              push           zlb
              add            %rcx, zlb
              push           %rcx
reczloop:     shr            N
              shr            N2
              call           next_level
              shl            N2
              shl            N
              pop            %rcx
              inc            %rcx
              pop            zlb
              cmp            $2, %rcx
              jne            reczloop
              pop            %rbx
              inc            %rbx
              pop            ylb
              cmp            $2, %rbx
              jne            recyloop
              pop            %rax
              inc            %rax
              pop            xlb
              cmp            $2, %rax
              jne            recxloop
              mov            xlb, %rax
              mov            ylb, %rbx
              mov            zlb, %rcx
              add            N2, %rax
              add            N2, %rbx
              add            N2, %rcx
              mov            %rax, xub
              mov            %rbx, yub
              mov            %rcx, zub
              bsr            N, %rax
              mov            Wrptr, %rbx
              mov            Wiptr, %rcx
              lea            (%rbx, %rax, 8), %rdx
              lea            (%rcx, %rax, 8), %rsi
              mov            %rdx, Wr
              mov            %rsi, Wi
              mov            N2, %rax
              imul           N0, %rax
              mov            %rax, N2N0
              imul           N0, %rax
              mov            %rax, N2N0N0
              mov            xlb, kx
kxloop:       mov            Wr, %rax
              mov            Wi, %rdx
              mov            kx, %rcx
              vbroadcastsd   (%rax, %rcx, 8), %ymm0
              vbroadcastsd   (%rdx, %rcx, 8), %ymm1
              vmovupd        %ymm0, COS100
              vmovupd        %ymm1, SIN100
              mov            ylb, ky
kyloop:       mov            Wr, %rax
              mov            Wi, %rdx
              vbroadcastsd   (%rax, ky, 8), %ymm0
              vbroadcastsd   (%rdx, ky, 8), %ymm1
              vmovupd        %ymm0, COS010
              vmovupd        %ymm1, SIN010
              vmulpd         SIN100, %ymm1, %ymm3
              vmulpd         COS100, %ymm1, %ymm4
              vfmsub231pd    COS100, %ymm0, %ymm3
              vfmadd231pd    SIN100, %ymm0, %ymm4
              vmovupd        %ymm3, COS110
              vmovupd        %ymm4, SIN110
              mov            zlb, kz
kzloop:       mov            Wr, %rax
              mov            Wi, %rdx
              vmovapd        (%rax, kz, 8), %ymm0
              vmovapd        (%rdx, kz, 8), %ymm1
              vmovupd        %ymm0, COS001
              vmovupd        %ymm1, SIN001
              vmulpd         SIN100, %ymm1, %ymm3
              vmulpd         COS100, %ymm1, %ymm4
              vfmsub231pd    COS100, %ymm0, %ymm3
              vfmadd231pd    SIN100, %ymm0, %ymm4
              vmulpd         SIN010, %ymm1, %ymm5
              vmulpd         COS010, %ymm1, %ymm6
              vfmsub231pd    COS010, %ymm0, %ymm5
              vfmadd231pd    SIN010, %ymm0, %ymm6
              vmulpd         SIN110, %ymm1, %ymm7
              vmulpd         COS110, %ymm1, %ymm8
              vfmsub231pd    COS110, %ymm0, %ymm7
              vfmadd231pd    SIN110, %ymm0, %ymm8
              vmovupd        %ymm3, COS101
              vmovupd        %ymm4, SIN101
              vmovupd        %ymm5, COS011
              vmovupd        %ymm6, SIN011
              vmovupd        %ymm7, COS111
              vmovupd        %ymm8, SIN111
              mov            kx, %rax
              imul           N0, %rax
              add            ky, %rax
              imul           N0, %rax
              add            kz, %rax
              mov            %rax, %rbx
              mov            %rax, %rcx
              mov            %rax, %rdx
              add            N2, %rbx
              add            N2, %rdx
              add            N2N0, %rcx
              add            N2N0, %rdx
              vmovapd        (X, %rax, 8), TR000
              vmovapd        (Y, %rax, 8), TI000
              vmovapd        (X, %rbx, 8), ER100
              vmovapd        (Y, %rbx, 8), EI100
              vmovapd        (X, %rcx, 8), ER010
              vmovapd        (Y, %rcx, 8), EI010
              vmovapd        (X, %rdx, 8), ER110
              vmovapd        (Y, %rdx, 8), EI110
              vmovupd        TR000, ER000
              vmovupd        TI000, EI000
              push           %rax
              push           %rbx
              push           %rcx
              push           %rdx
              add            N2N0N0, %rax
              add            N2N0N0, %rbx
              add            N2N0N0, %rcx
              add            N2N0N0, %rdx
              vmovapd        (X, %rax, 8), ER001
              vmovapd        (Y, %rax, 8), EI001
              vmovapd        (X, %rbx, 8), ER101
              vmovapd        (Y, %rbx, 8), EI101
              vmovapd        (X, %rcx, 8), ER011
              vmovapd        (Y, %rcx, 8), EI011
              vmovapd        (X, %rdx, 8), ER111
              vmovapd        (Y, %rdx, 8), EI111
              vmovapd        ER100, TR000
              vmovapd        EI100, TI000
              vmulpd         SIN100, TI000, ER100
              vmulpd         COS100, TI000, EI100
              vfmsub231pd    COS100, TR000, ER100
              vfmadd231pd    SIN100, TR000, EI100
              vmovapd        ER010, TR000
              vmovapd        EI010, TI000
              vmulpd         SIN010, TI000, ER010
              vmulpd         COS010, TI000, EI010
              vfmsub231pd    COS010, TR000, ER010
              vfmadd231pd    SIN010, TR000, EI010
              vmovapd        ER001, TR000
              vmovapd        EI001, TI000
              vmulpd         SIN001, TI000, ER001
              vmulpd         COS001, TI000, EI001
              vfmsub231pd    COS001, TR000, ER001
              vfmadd231pd    SIN001, TR000, EI001
              vmovapd        ER110, TR000
              vmovapd        EI110, TI000
              vmulpd         SIN110, TI000, ER110
              vmulpd         COS110, TI000, EI110
              vfmsub231pd    COS110, TR000, ER110
              vfmadd231pd    SIN110, TR000, EI110
              vmovapd        ER101, TR000
              vmovapd        EI101, TI000
              vmulpd         SIN101, TI000, ER101
              vmulpd         COS101, TI000, EI101
              vfmsub231pd    COS101, TR000, ER101
              vfmadd231pd    SIN101, TR000, EI101
              vmovapd        ER011, TR000
              vmovapd        EI011, TI000
              vmulpd         SIN011, TI000, ER011
              vmulpd         COS011, TI000, EI011
              vfmsub231pd    COS011, TR000, ER011
              vfmadd231pd    SIN011, TR000, EI011
              vmovapd        ER111, TR000
              vmovapd        EI111, TI000
              vmulpd         SIN111, TI000, ER111
              vmulpd         COS111, TI000, EI111
              vfmsub231pd    COS111, TR000, ER111
              vfmadd231pd    SIN111, TR000, EI111
              vaddpd         ER000, ER001, TR000
              vaddpd         EI000, EI001, TI000
              vsubpd         ER000, ER001, ER001
              vsubpd         EI000, EI001, EI001
              vmulpd         NONE, ER001, ER001
              vmulpd         NONE, EI001, EI001
              vmovupd        TR000, ER000
              vmovupd        TI000, EI000
              vmovapd        ER100, TR000
              vmovapd        EI100, TI000
              vaddpd         ER101, ER100, ER100
              vaddpd         EI101, EI100, EI100
              vsubpd         ER101, TR000, ER101
              vsubpd         EI101, TI000, EI101
              vmovapd        ER010, TR000
              vmovapd        EI010, TI000
              vaddpd         ER011, ER010, ER010
              vaddpd         EI011, EI010, EI010
              vsubpd         ER011, TR000, ER011
              vsubpd         EI011, TI000, EI011
              vmovapd        ER110, TR000
              vmovapd        EI110, TI000
              vaddpd         ER111, ER010, ER110
              vaddpd         EI111, EI010, EI110
              vsubpd         ER111, TR000, ER111
              vsubpd         EI111, TI000, EI111
              vaddpd         ER000, ER010, TR000
              vaddpd         EI000, EI010, TI000
              vsubpd         ER000, ER010, ER010
              vsubpd         EI000, EI010, EI010
              vmulpd         NONE, ER010, ER010
              vmulpd         NONE, EI010, EI010
              vmovupd        TR000, ER000
              vmovupd        TI000, EI000
              vmovapd        ER100, TR000
              vmovapd        EI100, TI000
              vaddpd         ER110, ER100, ER100
              vaddpd         EI110, EI100, EI100
              vsubpd         ER110, TR000, ER110
              vsubpd         EI110, TI000, EI110
              vmovapd        ER001, TR000
              vmovapd        EI001, TI000
              vaddpd         ER011, ER001, ER001
              vaddpd         EI011, EI001, EI001
              vsubpd         ER011, TR000, ER011
              vsubpd         EI011, TI000, EI011
              vmovapd        ER101, TR000
              vmovapd        EI101, TI000
              vaddpd         ER111, ER101, ER101
              vaddpd         EI111, EI101, EI101
              vsubpd         ER111, TR000, ER111
              vsubpd         EI111, TI000, EI111
              vaddpd         ER000, ER100, TR000
              vaddpd         EI000, EI100, TI000
              vsubpd         ER000, ER100, ER100
              vsubpd         EI000, EI100, EI100
              vmulpd         NONE, ER100, ER100
              vmulpd         NONE, EI100, EI100
              vmovupd        TR000, ER000
              vmovupd        TI000, EI000
              vmovapd        ER001, TR000
              vmovapd        EI001, TI000
              vaddpd         ER101, ER001, ER001
              vaddpd         EI101, EI001, EI001
              vsubpd         ER101, TR000, ER101
              vsubpd         EI101, TI000, EI101
              vmovapd        ER010, TR000
              vmovapd        EI010, TI000
              vaddpd         ER110, ER010, ER010
              vaddpd         EI110, EI010, EI010
              vsubpd         ER110, TR000, ER110
              vsubpd         EI110, TI000, EI110
              vmovapd        ER011, TR000
              vmovapd        EI011, TI000
              vaddpd         ER111, ER011, ER011
              vaddpd         EI111, EI011, EI011
              vsubpd         ER111, TR000, ER111
              vsubpd         EI111, TI000, EI111
              pop            %rdx
              pop            %rcx
              pop            %rbx
              pop            %rax
              vmovupd        ER000, TR000
              vmovupd        EI000, TI000
              vmovapd        TR000, (X, %rax, 8)
              vmovapd        TI000, (Y, %rax, 8)
              vmovapd        ER100, (X, %rbx, 8)
              vmovapd        EI100, (Y, %rbx, 8)
              vmovapd        ER010, (X, %rcx, 8)
              vmovapd        EI010, (Y, %rcx, 8)
              vmovapd        ER110, (X, %rdx, 8)
              vmovapd        EI110, (Y, %rdx, 8)
              add            N2N0N0, %rax
              add            N2N0N0, %rbx
              add            N2N0N0, %rcx
              add            N2N0N0, %rdx
              vmovapd        ER001, (X, %rax, 8)
              vmovapd        EI001, (Y, %rax, 8)
              vmovapd        ER101, (X, %rbx, 8)
              vmovapd        EI101, (Y, %rbx, 8)
              vmovapd        ER011, (X, %rcx, 8)
              vmovapd        EI011, (Y, %rcx, 8)
              vmovapd        ER111, (X, %rdx, 8)
              vmovapd        EI111, (Y, %rdx, 8)
              add            $4, kz
              cmp            zlb, kz
              jne            kzloop
              inc            ky
              cmp            ylb, ky
              jne            kyloop
              inc            kx
              cmp            xub, kx
              jne            kxloop
done:         ret
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
