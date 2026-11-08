.intel_syntax noprefix
.text
# void decoder_layer(float *x, void **weights, void **state, float *att_scratch,
#                    float *mlp_scratch, float *work)
# weights: pre_norm, post_norm, attention weights, mlp gate/up/down, type
# type = 0: DeltaNet, 1: full attention; work contains 2048 floats.
.globl decoder_layer
.type decoder_layer,@function
decoder_layer:
    push rbp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 8
    mov rbx, rdi
    mov r12, rsi
    mov r13, rdx
    mov r14, rcx
    mov r15, r8
    mov rbp, r9
    mov rdi, rbx
    mov rsi, [r12]
    mov rdx, rbp
    mov ecx, 1024
    vmovss xmm0, DWORD PTR [rip+.Ldec_eps]
    vmovss xmm1, DWORD PTR [rip+.Ldec_one]
    call rmsnorm_f32@PLT
    mov rdi, rbp
    lea rsi, [rbp+4096]
    mov rdx, [r12+16]
    mov rcx, r13
    mov r8, r14
    cmp QWORD PTR [r12+48], 0
    jne .Ldec_full
    call delta_layer@PLT
    jmp .Ldec_residual1
.Ldec_full:
    call full_attention_layer@PLT
.Ldec_residual1:
    xor eax, eax
.Ldec_add1:
    vmovss xmm0, DWORD PTR [rbx+rax*4]
    vaddss xmm0, xmm0, DWORD PTR [rbp+rax*4+4096]
    vmovss DWORD PTR [rbx+rax*4], xmm0
    inc eax
    cmp eax, 1024
    jb .Ldec_add1
    mov rdi, rbx
    mov rsi, [r12+8]
    mov rdx, rbp
    mov ecx, 1024
    vmovss xmm0, DWORD PTR [rip+.Ldec_eps]
    vmovss xmm1, DWORD PTR [rip+.Ldec_one]
    call rmsnorm_f32@PLT
    mov rdi, [r12+24]
    mov rsi, [r12+32]
    mov rdx, [r12+40]
    mov rcx, rbp
    lea r8, [rbp+4096]
    mov r9d, 1024
    sub rsp, 16
    mov QWORD PTR [rsp], 3584
    mov [rsp+8], r15
    call mlp_bf16@PLT
    add rsp, 16
    xor eax, eax
.Ldec_add2:
    vmovss xmm0, DWORD PTR [rbx+rax*4]
    vaddss xmm0, xmm0, DWORD PTR [rbp+rax*4+4096]
    vmovss DWORD PTR [rbx+rax*4], xmm0
    inc eax
    cmp eax, 1024
    jb .Ldec_add2
    add rsp, 8
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
.size decoder_layer,.-decoder_layer
.section .rodata
.Ldec_eps: .float 0.000001
.Ldec_one: .float 1.0
.section .note.GNU-stack,"",@progbits
