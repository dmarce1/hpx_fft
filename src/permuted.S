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
#define       Non            %rbx
#define       Pon            %r13
#define       Nsave          %r14
#define       Psave          %r15

              .global        permuted_increment

              .text

permuted_increment:
              push           %r12
              push           %r13
              push           %r14
              push           %r15
              push           %rbx
              mov            $1, RC
              xor            I, I
              mov            (Nptr), N
              mov            (Pptr), P
loop:         movzxb         (Indices, I), J
			  mov            N, Nsave
			  mov            P, Psave
			  mov            N, Non
			  mov            P, Pon
			  bts            I, Non
              bts            J, Pon
              btr            I, N
              btr            J, P
              bt             I, N
              cmovnc         Pon, P
              cmovnc         Non, N
              bt             I, Locked
              cmovc          Nsave, N
              cmovc          Psave, P
              bt             I, N
              jnc            done
              inc            I
              cmp            I, Sz
              jne            loop
done:         mov            N, (Nptr)
              mov            P, (Pptr)
              pop            %rbx
              pop            %r15
              pop            %r14
              pop            %r13
              pop            %r12
              ret
