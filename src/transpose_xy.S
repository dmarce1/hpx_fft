#define       X              %rdi
#define       N              %rsi
#define       M              %rdx
#define       Ix             %r8
#define       Iy             %r9
#define       dst            %r10
#define       src            %r11
#define       M0             $4

              .global        transpose_xy


              .text

transpose_xy: xor            Ix, Ix
              xor            Iy, Iy
              mov            N, M
              call           next_level
              ret
next_level:   cmp            Ix, Iy
              jl             done
              cmp            M0, M
              jg             next_levels
              imul           $2, N, %rax
              imul           $3, N, %rcx
              cmp            Ix, Iy
              je             on_diagonal
              mov            Ix, src
              mov            Iy, dst
              imul           N, src
              imul           N, dst
              add            Iy, src
              add            Ix, dst
              lea            (X, src, 8), src
              lea            (X, dst, 8), dst
              vmovapd        (src), %ymm0
              vmovapd        (src, N, 8), %ymm1
              vmovapd        (src, %rax, 8), %ymm2
              vmovapd        (src, %rcx, 8), %ymm3
              vmovapd        (dst), %ymm8
              vmovapd        (dst, N, 8), %ymm9
              vmovapd        (dst, %rax, 8), %ymm10
              vmovapd        (dst, %rcx, 8), %ymm11
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
              vmovapd        %ymm0, (dst)
              vmovapd        %ymm1, (dst, N, 8)
              vmovapd        %ymm2, (dst, %rax, 8)
              vmovapd        %ymm3, (dst, %rcx, 8)
              vmovapd        %ymm8, (src)
              vmovapd        %ymm9, (src, N, 8)
              vmovapd        %ymm10, (src, %rax, 8)
              vmovapd        %ymm11, (src, %rcx, 8)
              jmp            done
on_diagonal:  mov            Ix, src
              imul           N, src
              add            Ix, src
              lea            (X, src, 8), src
              vmovapd        (src), %ymm0
              vmovapd        (src, N, 8), %ymm1
              vmovapd        (src, %rax, 8), %ymm2
              vmovapd        (src, %rcx, 8), %ymm3
              vunpcklpd      %ymm1, %ymm0, %ymm4
              vunpckhpd      %ymm1, %ymm0, %ymm5
              vunpcklpd      %ymm3, %ymm2, %ymm6
              vunpckhpd      %ymm3, %ymm2, %ymm7
              vperm2f128     $0x20, %ymm6, %ymm4, %ymm0
              vperm2f128     $0x20, %ymm7, %ymm5, %ymm1
              vperm2f128     $0x31, %ymm6, %ymm4, %ymm2
              vperm2f128     $0x31, %ymm7, %ymm5, %ymm3
              vmovapd        %ymm0, (src)
              vmovapd        %ymm1, (src, N, 8)
              vmovapd        %ymm2, (src, %rax, 8)
              vmovapd        %ymm3, (src, %rcx, 8)
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





