.intel_syntax noprefix
.text
# uint32_t sample_topk(const float *logits, size_t n, size_t k,
#                      uint32_t random_bits, float temperature)
# 1 <= k <= 100, n > 0, temperature > 0. Static scratch, not thread-safe.
.globl sample_topk
.type sample_topk,@function
sample_topk:
    push rbx
    push r12
    push r13
    push r14
    push r15
    mov r12, rdi
    mov r13, rsi
    mov r14, rdx
    mov r15d, ecx
    vmovss DWORD PTR [rip+sample_temperature], xmm0
    mov QWORD PTR [rip+sample_count], 0
    xor ebx, ebx
.Lsample_scan:
    cmp rbx, r13
    jae .Lsample_weights
    vmovss xmm0, DWORD PTR [r12+rbx*4]
    mov rcx, [rip+sample_count]
    cmp rcx, r14
    jb .Lsample_find
    lea rax, [rip+sample_values]
    vucomiss xmm0, DWORD PTR [rax+rcx*4-4]
    jbe .Lsample_next
    dec rcx
.Lsample_find:
    lea rax, [rip+sample_values]
    lea rdx, [rip+sample_ids]
.Lsample_shift:
    test rcx, rcx
    jz .Lsample_place
    vucomiss xmm0, DWORD PTR [rax+rcx*4-4]
    jbe .Lsample_place
    vmovss xmm1, DWORD PTR [rax+rcx*4-4]
    vmovss DWORD PTR [rax+rcx*4], xmm1
    mov esi, DWORD PTR [rdx+rcx*4-4]
    mov DWORD PTR [rdx+rcx*4], esi
    dec rcx
    jmp .Lsample_shift
.Lsample_place:
    vmovss DWORD PTR [rax+rcx*4], xmm0
    mov DWORD PTR [rdx+rcx*4], ebx
    mov rax, [rip+sample_count]
    cmp rax, r14
    jae .Lsample_next
    inc QWORD PTR [rip+sample_count]
.Lsample_next:
    inc rbx
    jmp .Lsample_scan
.Lsample_weights:
    lea rax, [rip+sample_values]
    vmovss xmm8, DWORD PTR [rax]
    vxorps xmm9, xmm9, xmm9
    xor ebx, ebx
.Lsample_exp:
    cmp rbx, [rip+sample_count]
    jae .Lsample_draw
    lea rax, [rip+sample_values]
    vmovss xmm0, DWORD PTR [rax+rbx*4]
    vsubss xmm0, xmm0, xmm8
    vdivss xmm0, xmm0, DWORD PTR [rip+sample_temperature]
    vucomiss xmm0, DWORD PTR [rip+sample_min]
    jb .Lsample_zero
    call exp_f32@PLT
    jmp .Lsample_save
.Lsample_zero:
    vxorps xmm0, xmm0, xmm0
.Lsample_save:
    lea rax, [rip+sample_probs]
    vmovss DWORD PTR [rax+rbx*4], xmm0
    vaddss xmm9, xmm9, xmm0
    inc rbx
    jmp .Lsample_exp
.Lsample_draw:
    shr r15d, 8
    vcvtsi2ss xmm0, xmm0, r15d
    vmulss xmm0, xmm0, DWORD PTR [rip+sample_inverse_24]
    vmulss xmm0, xmm0, xmm9
    lea rax, [rip+sample_probs]
    xor ecx, ecx
.Lsample_choose:
    mov rdx, [rip+sample_count]
    dec rdx
    cmp rcx, rdx
    jae .Lsample_selected
    vsubss xmm0, xmm0, DWORD PTR [rax+rcx*4]
    vxorps xmm1, xmm1, xmm1
    vucomiss xmm0, xmm1
    jb .Lsample_selected
    inc rcx
    jmp .Lsample_choose
.Lsample_selected:
    lea rax, [rip+sample_ids]
    mov eax, DWORD PTR [rax+rcx*4]
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret
.size sample_topk,.-sample_topk
.section .rodata
sample_min: .float -80.0
sample_inverse_24: .float 0.000000059604644775390625
.bss
.align 16
sample_temperature: .skip 4
sample_count: .skip 8
sample_values: .skip 400
sample_ids: .skip 400
sample_probs: .skip 400
.section .note.GNU-stack,"",@progbits
