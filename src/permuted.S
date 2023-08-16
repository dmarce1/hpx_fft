#define       Nptr           %rdi
#define       Pptr           %rsi
#define       Locked         %rdx
#define       Indices        %rcx
#define       Sz             %r8
#define       N              %r9
#define       P              %r10
#define       I              %r11
#define       J              %r12
#define       RC             %rax

              .global        permuted_increment

              .text

permuted_increment:
              push           %r12
              mov            $1, RC
              xor            I, I
              mov            (Nptr), N
              mov            (Pptr), P
loop:         bt             I, Locked
              jc             skip
              movzxb         (Indices, I), J
              bt             I, N
              jnc            off
              btr            I, N
              btr            J, P
              jmp            skip
off:          bts            I, N
              bts            J, P
              xor            RC, RC
              jmp            done
skip:         inc            I
              cmp            I, Sz
              jne            loop
done:         mov            N, (Nptr)
              mov            P, (Pptr)
              pop            %r12
              ret
