.intel_syntax noprefix
.text
# int64_t model_step(size_t token, void **cfg, size_t produce_logits)
# cfg: embedding BF16, final norm FP32, 24 descriptor pointers, 24 state
# pointers, 24 attention-scratch pointers, MLP scratch, decoder work, x, logits.
# On prefill without logits returns -1; otherwise returns greedy token ID.
.globl model_step
.type model_step,@function
model_step:
    push rbx
    push r12
    push r13
    push r14
    push r15
    mov r12, rsi
    mov r13, rdx
    mov rax, rdi
    shl rax, 11
    mov rdi, [r12]
    add rdi, rax
    mov rsi, [r12+56]
    mov edx, 1024
    call bf16_to_f32@PLT
    xor ebx, ebx
.Lmodel_layer:
    cmp rbx, 24
    jae .Lmodel_logits
    mov rdi, [r12+56]
    mov rax, [r12+16]
    mov rsi, [rax+rbx*8]
    mov rax, [r12+24]
    mov rdx, [rax+rbx*8]
    mov rax, [r12+32]
    mov rcx, [rax+rbx*8]
    mov r8, [r12+40]
    mov r9, [r12+48]
    call decoder_layer@PLT
    inc rbx
    jmp .Lmodel_layer
.Lmodel_logits:
    test r13, r13
    jz .Lmodel_no_logits
    mov rdi, [r12+56]
    mov rsi, [r12+8]
    mov rdx, rdi
    mov ecx, 1024
    vmovss xmm0, DWORD PTR [rip+.Lmodel_eps]
    vmovss xmm1, DWORD PTR [rip+.Lmodel_one]
    call rmsnorm_f32@PLT
    mov rdi, [r12]
    mov rsi, [r12+56]
    mov rdx, [r12+64]
    mov ecx, 248320
    mov r8d, 1024
    call matvec_bf16_f32@PLT
    mov rdx, [r12+64]
    vmovss xmm0, DWORD PTR [rdx]
    xor eax, eax
    mov ecx, 1
.Lmodel_argmax:
    cmp ecx, 248320
    jae .Lmodel_done
    vmovss xmm1, DWORD PTR [rdx+rcx*4]
    vucomiss xmm1, xmm0
    jbe .Lmodel_next
    vmovss xmm0, xmm1, xmm1
    mov eax, ecx
.Lmodel_next:
    inc ecx
    jmp .Lmodel_argmax
.Lmodel_no_logits:
    mov rax, -1
.Lmodel_done:
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret
.size model_step,.-model_step
.section .rodata
.Lmodel_eps: .float 0.000001
.Lmodel_one: .float 1.0
.section .note.GNU-stack,"",@progbits
