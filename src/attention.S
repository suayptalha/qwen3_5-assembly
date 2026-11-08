.intel_syntax noprefix
.text
# void full_attention_layer(x, out, weights, state, scratch)
# weights: q/k/v/o BF16 matrix pointers, q/k norm FP32 pointers.
# state: k0/k1/v0/v1 cache pointers, cos/sin pointers, token count, scores pointer.
# scratch: 29696 bytes. The caller provides cos/sin for this position.
.globl full_attention_layer
.type full_attention_layer,@function
full_attention_layer:
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
    mov rdi, [r13]
    mov rsi, r12
    mov rdx, r15
    mov ecx, 4096
    mov r8d, 1024
    call matvec_bf16_f32@PLT
    mov rdi, [r13+8]
    mov rsi, r12
    lea rdx, [r15+16384]
    mov ecx, 512
    mov r8d, 1024
    call matvec_bf16_f32@PLT
    mov rdi, [r13+16]
    mov rsi, r12
    lea rdx, [r15+18432]
    mov ecx, 512
    mov r8d, 1024
    call matvec_bf16_f32@PLT
    mov QWORD PTR [rsp], 0
.Lkey_head:
    mov rax, [rsp]
    cmp rax, 2
    jae .Lquery_begin
    shl rax, 10
    lea rdi, [r15+rax+16384]
    mov rsi, [r13+40]
    mov rdx, rdi
    mov ecx, 256
    vmovss xmm0, DWORD PTR [rip+.Latt_eps]
    vmovss xmm1, DWORD PTR [rip+.Latt_one]
    call rmsnorm_f32@PLT
    mov rax, [rsp]
    shl rax, 10
    lea rdi, [r15+rax+16384]
    mov rsi, [r14+32]
    mov rdx, [r14+40]
    mov ecx, 32
    call rope_f32@PLT
    mov rax, [rsp]
    mov rdx, [r14+rax*8]
    mov rcx, [r14+48]
    dec rcx
    shl rcx, 10
    add rdx, rcx
    mov rcx, [r14+rax*8+16]
    mov r8, [r14+48]
    dec r8
    shl r8, 10
    add rcx, r8
    shl rax, 10
    lea rsi, [r15+rax+16384]
    lea rdi, [r15+rax+18432]
    xor eax, eax
.Lstore_kv:
    vmovss xmm0, DWORD PTR [rsi+rax*4]
    vmovss DWORD PTR [rdx+rax*4], xmm0
    vmovss xmm0, DWORD PTR [rdi+rax*4]
    vmovss DWORD PTR [rcx+rax*4], xmm0
    inc eax
    cmp eax, 256
    jb .Lstore_kv
    inc QWORD PTR [rsp]
    jmp .Lkey_head
.Lquery_begin:
    mov QWORD PTR [rsp], 0
.Lquery_head:
    mov rax, [rsp]
    cmp rax, 8
    jae .Loutput
    shl rax, 11
    lea rdi, [r15+rax]
    mov rsi, [r13+32]
    mov rdx, rdi
    mov ecx, 256
    vmovss xmm0, DWORD PTR [rip+.Latt_eps]
    vmovss xmm1, DWORD PTR [rip+.Latt_one]
    call rmsnorm_f32@PLT
    mov rax, [rsp]
    shl rax, 11
    lea rdi, [r15+rax]
    mov rsi, [r14+32]
    mov rdx, [r14+40]
    mov ecx, 32
    call rope_f32@PLT
    mov rax, [rsp]
    mov rdx, rax
    shr rdx, 2
    mov rsi, [r14+rdx*8]
    mov rdx, [r14+rdx*8+16]
    mov rcx, rax
    shl rcx, 10
    lea rcx, [r15+rcx+20480]
    shl rax, 11
    lea rdi, [r15+rax]
    mov r8, [r14+48]
    mov r9d, 256
    sub rsp, 16
    mov rax, [r14+56]
    mov [rsp], rax
    call attention_f32@PLT
    add rsp, 16
    mov rax, [rsp]
    shl rax, 11
    lea rdi, [r15+rax+1024]
    mov rsi, rdi
    mov edx, 256
    call sigmoid_f32@PLT
    mov rax, [rsp]
    mov rdx, rax
    shl rdx, 10
    lea rcx, [r15+rdx+20480]
    shl rax, 11
    lea rdx, [r15+rax+1024]
    xor eax, eax
.Lgate_att:
    vmovss xmm0, DWORD PTR [rcx+rax*4]
    vmulss xmm0, xmm0, DWORD PTR [rdx+rax*4]
    vmovss DWORD PTR [rcx+rax*4], xmm0
    inc eax
    cmp eax, 256
    jb .Lgate_att
    inc QWORD PTR [rsp]
    jmp .Lquery_head
.Loutput:
    mov rdi, [r13+24]
    lea rsi, [r15+20480]
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
.size full_attention_layer,.-full_attention_layer
.section .rodata
.Latt_eps: .float 0.000001
.Latt_one: .float 1.0
.section .note.GNU-stack,"",@progbits
