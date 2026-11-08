.intel_syntax noprefix
.text
# Fixed Qwen3.5-0.8B single-token DeltaNet layer. See tests/test_real_deltanet.py
# for the pointer-only weight/state structures and scratch size.
# void delta_layer(const float *x, float *out, const void **weights,
#                  void **state, float *scratch)
.globl delta_layer
.type delta_layer,@function
delta_layer:
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 16
    mov r12, rdi
    mov rbx, rsi
    mov r13, rdx
    mov r14, rcx
    mov r15, r8
    # qkv, z, beta, a projections
    mov rdi, [r13]
    mov rsi, r12
    mov rdx, r15
    mov ecx, 6144
    mov r8d, 1024
    call matvec_bf16_f32@PLT
    mov rdi, [r13+8]
    mov rsi, r12
    lea rdx, [r15+24576]
    mov ecx, 2048
    mov r8d, 1024
    call matvec_bf16_f32@PLT
    mov rdi, [r13+16]
    mov rsi, r12
    lea rdx, [r15+32768]
    mov ecx, 16
    mov r8d, 1024
    call matvec_bf16_f32@PLT
    lea rdi, [r15+32768]
    mov rsi, rdi
    mov edx, 16
    call sigmoid_f32@PLT
    mov rdi, [r13+24]
    mov rsi, r12
    lea rdx, [r15+32832]
    mov ecx, 16
    mov r8d, 1024
    call matvec_bf16_f32@PLT
    xor eax, eax
    mov rdx, [r13+48]
.Ladd_dt:
    vmovss xmm0, DWORD PTR [r15+rax*4+32832]
    vaddss xmm0, xmm0, DWORD PTR [rdx+rax*4]
    vmovss DWORD PTR [r15+rax*4+32832], xmm0
    inc eax
    cmp eax, 16
    jb .Ladd_dt
    lea rdi, [r15+32832]
    mov rsi, rdi
    mov edx, 16
    call softplus_f32@PLT
    mov QWORD PTR [rsp], 0
.Ldecay_loop:
    mov rax, [rsp]
    cmp rax, 16
    jae .Ldecay_done
    mov rdx, [r13+40]
    vmovss xmm0, DWORD PTR [rdx+rax*4]
    call exp_f32@PLT
    mov rax, [rsp]
    vmulss xmm0, xmm0, DWORD PTR [r15+rax*4+32832]
    vxorps xmm0, xmm0, XMMWORD PTR [rip+.Lsign]
    vucomiss xmm0, DWORD PTR [rip+.Lmin_exp]
    jb .Ldecay_zero
    call exp_f32@PLT
    jmp .Ldecay_store
.Ldecay_zero:
    vxorps xmm0, xmm0, xmm0
.Ldecay_store:
    mov rax, [rsp]
    vmovss DWORD PTR [r15+rax*4+32832], xmm0
    inc QWORD PTR [rsp]
    jmp .Ldecay_loop
.Ldecay_done:
    # causal conv and SiLU on qkv
    mov rdi, [r14]
    mov rsi, [r13+32]
    mov rdx, r15
    lea rcx, [r15+32896]
    mov r8d, 6144
    call causal_conv4_bf16@PLT
    lea rdi, [r15+32896]
    mov rsi, rdi
    mov edx, 6144
    call silu_f32@PLT
    mov QWORD PTR [rsp], 0
.Lhead:
    mov rax, [rsp]
    cmp rax, 16
    jae .Lprojection
    shl rax, 9
    lea rdi, [r15+rax+32896]
    lea rsi, [r15+65664]
    mov edx, 128
    vmovss xmm0, DWORD PTR [rip+.Leps]
    vmovss xmm1, DWORD PTR [rip+.Lscale]
    call l2norm_f32@PLT
    mov rax, [rsp]
    shl rax, 9
    lea rdi, [r15+rax+41088]
    lea rsi, [r15+66176]
    mov edx, 128
    vmovss xmm0, DWORD PTR [rip+.Leps]
    vmovss xmm1, DWORD PTR [rip+.Lone]
    call l2norm_f32@PLT
    mov rax, [rsp]
    mov rdi, [r14+8]
    mov rdx, rax
    shl rdx, 16
    add rdi, rdx
    lea rsi, [r15+65664]
    lea rdx, [r15+66176]
    mov rcx, rax
    shl rcx, 9
    lea r8, [r15+rcx+57472]
    lea rcx, [r15+rcx+49280]
    mov r9d, 128
    vmovss xmm0, DWORD PTR [r15+rax*4+32832]
    vmovss xmm1, DWORD PTR [r15+rax*4+32768]
    sub rsp, 16
    mov QWORD PTR [rsp], 128
    call delta_step@PLT
    add rsp, 16
    mov rax, [rsp]
    shl rax, 9
    lea rdi, [r15+rax+57472]
    mov rsi, [r13+56]
    lea rdx, [r15+66688]
    mov ecx, 128
    vmovss xmm0, DWORD PTR [rip+.Leps]
    vxorps xmm1, xmm1, xmm1
    call rmsnorm_f32@PLT
    mov rax, [rsp]
    shl rax, 9
    lea rdi, [r15+rax+24576]
    mov rsi, rdi
    mov edx, 128
    call silu_f32@PLT
    mov rax, [rsp]
    shl rax, 9
    lea rdx, [r15+rax+57472]
    lea rcx, [r15+rax+24576]
    xor eax, eax
.Lgate:
    vmovss xmm0, DWORD PTR [r15+rax*4+66688]
    vmulss xmm0, xmm0, DWORD PTR [rcx+rax*4]
    vmovss DWORD PTR [rdx+rax*4], xmm0
    inc eax
    cmp eax, 128
    jb .Lgate
    inc QWORD PTR [rsp]
    jmp .Lhead
.Lprojection:
    mov rdi, [r13+64]
    lea rsi, [r15+57472]
    mov rdx, rbx
    mov ecx, 1024
    mov r8d, 2048
    call matvec_bf16_f32@PLT
    add rsp, 16
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret
.size delta_layer,.-delta_layer
.section .rodata
.align 16
.Lsign: .long 0x80000000,0,0,0
.Lmin_exp: .float -80.0
.Leps: .float 0.000001
.Lscale: .float 0.08838834764831845
.Lone: .float 1.0
.section .note.GNU-stack,"",@progbits
