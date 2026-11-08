.intel_syntax noprefix
.text
# float dot_f32(const float *x, const float *y, size_t n)
.globl dot_f32
.type dot_f32,@function
dot_f32:
    vxorps ymm0, ymm0, ymm0
    xor eax, eax
.Lf32_loop:
    lea rcx, [rax+8]
    cmp rcx, rdx
    ja .Lf32_reduce
    vmovups ymm1, [rdi+rax*4]
    vfmadd231ps ymm0, ymm1, [rsi+rax*4]
    add rax, 8
    jmp .Lf32_loop
.Lf32_reduce:
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
.Lf32_tail:
    cmp rax, rdx
    jae .Lf32_done
    vmovss xmm1, DWORD PTR [rdi+rax*4]
    vmulss xmm1, xmm1, DWORD PTR [rsi+rax*4]
    vaddss xmm0, xmm0, xmm1
    inc rax
    jmp .Lf32_tail
.Lf32_done:
    vzeroupper
    ret
.size dot_f32,.-dot_f32

# float dot_bf16_f32(const uint16_t *weights, const float *x, size_t n)
.globl dot_bf16_f32
.type dot_bf16_f32,@function
dot_bf16_f32:
    vxorps ymm0, ymm0, ymm0
    xor eax, eax
.Lbf_loop:
    lea rcx, [rax+8]
    cmp rcx, rdx
    ja .Lbf_reduce
    vpmovzxwd ymm1, XMMWORD PTR [rdi+rax*2]
    vpslld ymm1, ymm1, 16
    vfmadd231ps ymm0, ymm1, [rsi+rax*4]
    add rax, 8
    jmp .Lbf_loop
.Lbf_reduce:
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
.Lbf_tail:
    cmp rax, rdx
    jae .Lbf_done
    movzx ecx, WORD PTR [rdi+rax*2]
    shl ecx, 16
    vmovd xmm1, ecx
    vmulss xmm1, xmm1, DWORD PTR [rsi+rax*4]
    vaddss xmm0, xmm0, xmm1
    inc rax
    jmp .Lbf_tail
.Lbf_done:
    vzeroupper
    ret
.size dot_bf16_f32,.-dot_bf16_f32
# void matvec_bf16_f32(const uint16_t *matrix, const float *x,
#                      float *out, size_t rows, size_t cols)
.globl matvec_bf16_f32
.type matvec_bf16_f32,@function
matvec_bf16_f32:
    push rbx
    push r12
    push r13
    push r14
    push r15
    mov r12, rdi
    mov r13, rsi
    mov r14, rdx
    mov r15, rcx
    mov rbx, r8
.Lmat_loop:
    test r15, r15
    jz .Lmat_done
    mov rdi, r12
    mov rsi, r13
    mov rdx, rbx
    call dot_bf16_f32@PLT
    vmovss DWORD PTR [r14], xmm0
    lea r12, [r12+rbx*2]
    add r14, 4
    dec r15
    jmp .Lmat_loop
.Lmat_done:
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret
.size matvec_bf16_f32,.-matvec_bf16_f32
# One head of the recurrent gated delta rule, in-place state[kdim, vdim].
# void delta_step(float *state, const float *q, const float *k, const float *v,
#                 float *out, size_t kdim, size_t vdim, float decay, float beta)
# q includes the 1/sqrt(kdim) scaling. q and k are already normalized.
.globl delta_step
.type delta_step,@function
delta_step:
    push rbx
    push r12
    push r13
    push r14
    push r15
    mov r10, QWORD PTR [rsp+48]  # seventh integer argument vdim
    xor r11d, r11d               # value index
.Ldelta_v:
    cmp r11, r10
    jae .Ldelta_done
    vxorps xmm2, xmm2, xmm2      # key dot decayed state
    xor eax, eax
.Ldelta_read:
    cmp rax, r9
    jae .Ldelta_update_start
    mov rbx, rax
    imul rbx, r10
    add rbx, r11
    vmovss xmm3, DWORD PTR [rdi+rbx*4]
    vmulss xmm3, xmm3, xmm0
    vmovss DWORD PTR [rdi+rbx*4], xmm3
    vmulss xmm3, xmm3, DWORD PTR [rdx+rax*4]
    vaddss xmm2, xmm2, xmm3
    inc rax
    jmp .Ldelta_read
.Ldelta_update_start:
    vmovss xmm4, DWORD PTR [rcx+r11*4]
    vsubss xmm4, xmm4, xmm2
    vmulss xmm4, xmm4, xmm1      # delta = beta * (value - memory)
    vxorps xmm5, xmm5, xmm5
    xor eax, eax
.Ldelta_update:
    cmp rax, r9
    jae .Ldelta_write
    mov rbx, rax
    imul rbx, r10
    add rbx, r11
    vmovss xmm3, DWORD PTR [rdx+rax*4]
    vmulss xmm3, xmm3, xmm4
    vaddss xmm3, xmm3, DWORD PTR [rdi+rbx*4]
    vmovss DWORD PTR [rdi+rbx*4], xmm3
    vmulss xmm3, xmm3, DWORD PTR [rsi+rax*4]
    vaddss xmm5, xmm5, xmm3
    inc rax
    jmp .Ldelta_update
.Ldelta_write:
    vmovss DWORD PTR [r8+r11*4], xmm5
    inc r11
    jmp .Ldelta_v
.Ldelta_done:
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret
.size delta_step,.-delta_step
# void rmsnorm_f32(const float *x, const float *weight, float *out,
#                  size_t n, float eps, float weight_offset)
.globl rmsnorm_f32
.type rmsnorm_f32,@function
rmsnorm_f32:
    test rcx, rcx
    jz .Lnorm_done
    vxorps xmm2, xmm2, xmm2
    xor eax, eax
.Lnorm_sum:
    vmovss xmm3, DWORD PTR [rdi+rax*4]
    vmulss xmm3, xmm3, xmm3
    vaddss xmm2, xmm2, xmm3
    inc rax
    cmp rax, rcx
    jb .Lnorm_sum
    vcvtsi2ss xmm3, xmm3, rcx
    vdivss xmm2, xmm2, xmm3
    vaddss xmm2, xmm2, xmm0
    vsqrtss xmm2, xmm2, xmm2
    xor eax, eax
.Lnorm_apply:
    vmovss xmm3, DWORD PTR [rdi+rax*4]
    vdivss xmm3, xmm3, xmm2
    vmovss xmm4, DWORD PTR [rsi+rax*4]
    vaddss xmm4, xmm4, xmm1
    vmulss xmm3, xmm3, xmm4
    vmovss DWORD PTR [rdx+rax*4], xmm3
    inc rax
    cmp rax, rcx
    jb .Lnorm_apply
.Lnorm_done:
    ret
.size rmsnorm_f32,.-rmsnorm_f32
# void rope_f32(float *x, const float *cos, const float *sin, size_t half)
# Rotates the first 2*half entries, pairing first and second halves.
.globl rope_f32
.type rope_f32,@function
rope_f32:
    xor eax, eax
.Lrope_loop:
    cmp rax, rcx
    jae .Lrope_done
    vmovss xmm0, DWORD PTR [rdi+rax*4]
    lea r8, [rax+rcx]
    vmovss xmm1, DWORD PTR [rdi+r8*4]
    vmovss xmm2, DWORD PTR [rsi+rax*4]
    vmovss xmm3, DWORD PTR [rdx+rax*4]
    vmulss xmm4, xmm0, xmm2
    vmulss xmm5, xmm1, xmm3
    vsubss xmm4, xmm4, xmm5
    vmulss xmm5, xmm1, xmm2
    vmulss xmm6, xmm0, xmm3
    vaddss xmm5, xmm5, xmm6
    vmovss DWORD PTR [rdi+rax*4], xmm4
    vmovss DWORD PTR [rdi+r8*4], xmm5
    inc rax
    jmp .Lrope_loop
.Lrope_done:
    ret
.size rope_f32,.-rope_f32
# void l2norm_f32(const float *x, float *out, size_t n, float eps, float scale)
.globl l2norm_f32
.type l2norm_f32,@function
l2norm_f32:
    vxorps xmm2, xmm2, xmm2
    xor eax, eax
.Ll2_sum:
    cmp rax, rdx
    jae .Ll2_apply_start
    vmovss xmm3, DWORD PTR [rdi+rax*4]
    vmulss xmm3, xmm3, xmm3
    vaddss xmm2, xmm2, xmm3
    inc rax
    jmp .Ll2_sum
.Ll2_apply_start:
    vaddss xmm2, xmm2, xmm0
    vsqrtss xmm2, xmm2, xmm2
    vdivss xmm2, xmm1, xmm2
    xor eax, eax
.Ll2_apply:
    cmp rax, rdx
    jae .Ll2_done
    vmovss xmm3, DWORD PTR [rdi+rax*4]
    vmulss xmm3, xmm3, xmm2
    vmovss DWORD PTR [rsi+rax*4], xmm3
    inc rax
    jmp .Ll2_apply
.Ll2_done:
    ret
.size l2norm_f32,.-l2norm_f32

# void attention_f32(q, keys, values, out, size_t tokens, size_t dim, scratch)
# keys/values are token-major; scratch has tokens floats. Non-empty context.
.globl attention_f32
.type attention_f32,@function
attention_f32:
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 16
    mov r12, rdi
    mov r13, rsi
    mov r14, rdx
    mov r15, rcx
    mov rbx, r8
    mov [rsp], r9
    mov rax, [rsp+64]
    mov [rsp+8], rax
    test rbx, rbx
    jz .Lattn_zero
    xor eax, eax
    mov [rsp+8], rax             # temporarily use slot as token counter
    mov rax, [rsp+64]
    # Scratch pointer is stored in the original stack argument.
.Lattn_score:
    mov rax, [rsp+8]
    cmp rax, rbx
    jae .Lattn_softmax
    imul rax, [rsp]
    lea rsi, [r13+rax*4]
    mov rdi, r12
    mov rdx, [rsp]
    call dot_f32@PLT
    vcvtsi2ss xmm1, xmm1, QWORD PTR [rsp]
    vsqrtss xmm1, xmm1, xmm1
    vdivss xmm0, xmm0, xmm1
    mov rax, [rsp+8]
    mov rdx, [rsp+64]
    vmovss DWORD PTR [rdx+rax*4], xmm0
    inc QWORD PTR [rsp+8]
    jmp .Lattn_score
.Lattn_softmax:
    mov rdi, [rsp+64]
    mov rsi, rdi
    mov rdx, rbx
    call softmax_f32@PLT
.Lattn_zero:
    xor eax, eax
.Lattn_zero_loop:
    cmp rax, [rsp]
    jae .Lattn_accum_start
    mov DWORD PTR [r15+rax*4], 0
    inc rax
    jmp .Lattn_zero_loop
.Lattn_accum_start:
    xor r8d, r8d
.Lattn_token:
    cmp r8, rbx
    jae .Lattn_done
    mov rax, r8
    imul rax, [rsp]
    lea rsi, [r14+rax*4]
    mov rax, [rsp+64]
    vmovss xmm0, DWORD PTR [rax+r8*4]
    xor ecx, ecx
.Lattn_dim:
    cmp rcx, [rsp]
    jae .Lattn_next
    vmovss xmm1, DWORD PTR [rsi+rcx*4]
    vmulss xmm1, xmm1, xmm0
    vaddss xmm1, xmm1, DWORD PTR [r15+rcx*4]
    vmovss DWORD PTR [r15+rcx*4], xmm1
    inc rcx
    jmp .Lattn_dim
.Lattn_next:
    inc r8
    jmp .Lattn_token
.Lattn_done:
    add rsp, 16
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret
.size attention_f32,.-attention_f32

# void bf16_to_f32(const uint16_t *input, float *output, size_t count)
.globl bf16_to_f32
.type bf16_to_f32,@function
bf16_to_f32:
    xor eax, eax
.Lconvert_vec:
    lea rcx, [rax+8]
    cmp rcx, rdx
    ja .Lconvert_tail
    vpmovzxwd ymm0, XMMWORD PTR [rdi+rax*2]
    vpslld ymm0, ymm0, 16
    vmovups YMMWORD PTR [rsi+rax*4], ymm0
    add rax, 8
    jmp .Lconvert_vec
.Lconvert_tail:
    cmp rax, rdx
    jae .Lconvert_done
    movzx ecx, WORD PTR [rdi+rax*2]
    shl ecx, 16
    mov DWORD PTR [rsi+rax*4], ecx
    inc rax
    jmp .Lconvert_tail
.Lconvert_done:
    vzeroupper
    ret
.size bf16_to_f32,.-bf16_to_f32

# void mlp_bf16(const uint16_t *gate, const uint16_t *up, const uint16_t *down,
#               const float *x, float *out, size_t hidden, size_t intermediate,
#               float *scratch) -- scratch has 2*intermediate floats.
.globl mlp_bf16
.type mlp_bf16,@function
mlp_bf16:
    push rbp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 40
    mov r12, rdi
    mov r13, rsi
    mov r14, rdx
    mov r15, rcx
    mov rbx, r8
    mov rbp, r9
    mov rax, [rsp+96]
    mov [rsp+8], rax             # intermediate
    mov rax, [rsp+104]
    mov [rsp], rax               # scratch
    mov rdi, r12
    mov rsi, r15
    mov rdx, rax
    mov rcx, [rsp+8]
    mov r8, rbp
    call matvec_bf16_f32@PLT
    mov rdi, [rsp]
    mov rsi, rdi
    mov rdx, [rsp+8]
    call silu_f32@PLT
    mov rdi, r13
    mov rsi, r15
    mov rdx, [rsp]
    mov rax, [rsp+8]
    lea rdx, [rdx+rax*4]
    mov rcx, rax
    mov r8, rbp
    call matvec_bf16_f32@PLT
    mov rdi, [rsp]
    mov rcx, [rsp+8]
    lea rsi, [rdi+rcx*4]
    xor eax, eax
.Lmlp_multiply:
    cmp rax, rcx
    jae .Lmlp_down
    vmovss xmm0, DWORD PTR [rdi+rax*4]
    vmulss xmm0, xmm0, DWORD PTR [rsi+rax*4]
    vmovss DWORD PTR [rdi+rax*4], xmm0
    inc rax
    jmp .Lmlp_multiply
.Lmlp_down:
    mov rdi, r14
    mov rsi, [rsp]
    mov rdx, rbx
    mov rcx, rbp
    mov r8, [rsp+8]
    call matvec_bf16_f32@PLT
    add rsp, 40
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
.size mlp_bf16,.-mlp_bf16

# void causal_conv4_bf16(float *state, const uint16_t *weights,
#                         const float *input, float *out, size_t channels)
# Four state and weight values per channel, chronological order. In-place state.
.globl causal_conv4_bf16
.type causal_conv4_bf16,@function
causal_conv4_bf16:
    xor eax, eax
.Lconv_channel:
    cmp rax, r8
    jae .Lconv_done
    mov r9, rax
    shl r9, 4
    lea r10, [rdi+r9]
    vmovss xmm0, DWORD PTR [r10+4]
    vmovss xmm1, DWORD PTR [r10+8]
    vmovss xmm2, DWORD PTR [r10+12]
    vmovss xmm3, DWORD PTR [rdx+rax*4]
    vmovss DWORD PTR [r10], xmm0
    vmovss DWORD PTR [r10+4], xmm1
    vmovss DWORD PTR [r10+8], xmm2
    vmovss DWORD PTR [r10+12], xmm3
    lea r9, [rax*8]
    movzx r11d, WORD PTR [rsi+r9]
    shl r11d, 16
    vmovd xmm4, r11d
    vmulss xmm0, xmm0, xmm4
    movzx r11d, WORD PTR [rsi+r9+2]
    shl r11d, 16
    vmovd xmm4, r11d
    vmulss xmm1, xmm1, xmm4
    vaddss xmm0, xmm0, xmm1
    movzx r11d, WORD PTR [rsi+r9+4]
    shl r11d, 16
    vmovd xmm4, r11d
    vmulss xmm2, xmm2, xmm4
    vaddss xmm0, xmm0, xmm2
    movzx r11d, WORD PTR [rsi+r9+6]
    shl r11d, 16
    vmovd xmm4, r11d
    vmulss xmm3, xmm3, xmm4
    vaddss xmm0, xmm0, xmm3
    vmovss DWORD PTR [rcx+rax*4], xmm0
    inc rax
    jmp .Lconv_channel
.Lconv_done:
    ret
.size causal_conv4_bf16,.-causal_conv4_bf16

# void softmax_f32(const float *input, float *output, size_t n)
# Supports finite input; output may alias input.
.globl softmax_f32
.type softmax_f32,@function
softmax_f32:
    test rdx, rdx
    jz .Lsoft_done
    push rbx
    push r12
    push r13
    push r14
    push r15
    mov r12, rdi
    mov r13, rsi
    mov r14, rdx
    vmovss xmm8, DWORD PTR [r12]
    mov eax, 1
.Lsoft_max:
    cmp rax, r14
    jae .Lsoft_exp_start
    vmaxss xmm8, xmm8, DWORD PTR [r12+rax*4]
    inc rax
    jmp .Lsoft_max
.Lsoft_exp_start:
    xor ebx, ebx
    vxorps xmm9, xmm9, xmm9
.Lsoft_exp:
    cmp rbx, r14
    jae .Lsoft_normalize_start
    vmovss xmm0, DWORD PTR [r12+rbx*4]
    vsubss xmm0, xmm0, xmm8
    vucomiss xmm0, DWORD PTR [rip+.Lminus80]
    jb .Lsoft_zero
    call exp_f32@PLT
    jmp .Lsoft_save
.Lsoft_zero:
    vxorps xmm0, xmm0, xmm0
.Lsoft_save:
    vmovss DWORD PTR [r13+rbx*4], xmm0
    vaddss xmm9, xmm9, xmm0
    inc rbx
    jmp .Lsoft_exp
.Lsoft_normalize_start:
    xor eax, eax
.Lsoft_normalize:
    cmp rax, r14
    jae .Lsoft_pops
    vmovss xmm0, DWORD PTR [r13+rax*4]
    vdivss xmm0, xmm0, xmm9
    vmovss DWORD PTR [r13+rax*4], xmm0
    inc rax
    jmp .Lsoft_normalize
.Lsoft_pops:
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
.Lsoft_done:
    ret
.size softmax_f32,.-softmax_f32

# void silu_f32(const float *input, float *output, size_t n)
# Numerically stable for finite inputs, in-place allowed.
.globl silu_f32
.type silu_f32,@function
silu_f32:
    push rbx
    push r12
    push r13
    mov r12, rdi
    mov r13, rsi
    mov rbx, rdx
    xor r10d, r10d
.Lsilu_loop:
    cmp r10, rbx
    jae .Lsilu_done
    vmovss xmm8, DWORD PTR [r12+r10*4]
    vxorps xmm9, xmm9, xmm9
    vucomiss xmm8, xmm9
    jb .Lsilu_negative
    vucomiss xmm8, DWORD PTR [rip+.Lplus80]
    ja .Lsilu_direct
    vxorps xmm0, xmm8, XMMWORD PTR [rip+.Lsignmask]
    call exp_f32@PLT
    vaddss xmm0, xmm0, DWORD PTR [rip+.Lone]
    vdivss xmm8, xmm8, xmm0
    jmp .Lsilu_direct
.Lsilu_negative:
    vucomiss xmm8, DWORD PTR [rip+.Lminus80]
    jb .Lsilu_zero
    vmovaps xmm0, xmm8
    call exp_f32@PLT
    vaddss xmm1, xmm0, DWORD PTR [rip+.Lone]
    vdivss xmm0, xmm0, xmm1
    vmulss xmm8, xmm8, xmm0
    jmp .Lsilu_direct
.Lsilu_zero:
    vxorps xmm8, xmm8, xmm8
.Lsilu_direct:
    vmovss DWORD PTR [r13+r10*4], xmm8
    inc r10
    jmp .Lsilu_loop
.Lsilu_done:
    pop r13
    pop r12
    pop rbx
    ret
.size silu_f32,.-silu_f32

# float log_f32(float x), x > 0 and finite, implemented with x87.
.globl log_f32
.type log_f32,@function
log_f32:
    sub rsp, 8
    vmovss DWORD PTR [rsp], xmm0
    fldln2
    fld DWORD PTR [rsp]
    fyl2x
    fstp DWORD PTR [rsp]
    vmovss xmm0, DWORD PTR [rsp]
    add rsp, 8
    ret
.size log_f32,.-log_f32

# void sigmoid_f32(const float *input, float *output, size_t n)
.globl sigmoid_f32
.type sigmoid_f32,@function
sigmoid_f32:
    push rbx
    push r12
    push r13
    mov r12, rdi
    mov r13, rsi
    mov rbx, rdx
    xor r10d, r10d
.Lsig_loop:
    cmp r10, rbx
    jae .Lsig_done
    vmovss xmm8, DWORD PTR [r12+r10*4]
    vxorps xmm9, xmm9, xmm9
    vucomiss xmm8, xmm9
    jb .Lsig_negative
    vucomiss xmm8, DWORD PTR [rip+.Lplus80]
    ja .Lsig_one
    vxorps xmm0, xmm8, XMMWORD PTR [rip+.Lsignmask]
    call exp_f32@PLT
    vaddss xmm0, xmm0, DWORD PTR [rip+.Lone]
    vmovss xmm1, DWORD PTR [rip+.Lone]
    vdivss xmm0, xmm1, xmm0
    jmp .Lsig_store
.Lsig_one:
    vmovss xmm0, DWORD PTR [rip+.Lone]
    jmp .Lsig_store
.Lsig_negative:
    vucomiss xmm8, DWORD PTR [rip+.Lminus80]
    jb .Lsig_zero
    vmovaps xmm0, xmm8
    call exp_f32@PLT
    vaddss xmm1, xmm0, DWORD PTR [rip+.Lone]
    vdivss xmm0, xmm0, xmm1
    jmp .Lsig_store
.Lsig_zero:
    vxorps xmm0, xmm0, xmm0
.Lsig_store:
    vmovss DWORD PTR [r13+r10*4], xmm0
    inc r10
    jmp .Lsig_loop
.Lsig_done:
    pop r13
    pop r12
    pop rbx
    ret
.size sigmoid_f32,.-sigmoid_f32

# void softplus_f32(const float *input, float *output, size_t n)
.globl softplus_f32
.type softplus_f32,@function
softplus_f32:
    push rbx
    push r12
    push r13
    mov r12, rdi
    mov r13, rsi
    mov rbx, rdx
    xor r10d, r10d
.Lsp_loop:
    cmp r10, rbx
    jae .Lsp_done
    vmovss xmm0, DWORD PTR [r12+r10*4]
    vucomiss xmm0, DWORD PTR [rip+.Lplus16]
    ja .Lsp_store
    vucomiss xmm0, DWORD PTR [rip+.Lminus80]
    jb .Lsp_zero
    vucomiss xmm0, DWORD PTR [rip+.Lminus16]
    jb .Lsp_small
    call exp_f32@PLT
    vaddss xmm0, xmm0, DWORD PTR [rip+.Lone]
    call log_f32@PLT
    jmp .Lsp_store
.Lsp_small:
    call exp_f32@PLT
    jmp .Lsp_store
.Lsp_zero:
    vxorps xmm0, xmm0, xmm0
.Lsp_store:
    vmovss DWORD PTR [r13+r10*4], xmm0
    inc r10
    jmp .Lsp_loop
.Lsp_done:
    pop r13
    pop r12
    pop rbx
    ret
.size softplus_f32,.-softplus_f32

# float exp_f32(float x): exp approximation for finite x in [-80, 80].
# Range reduction x=n*ln(2)+r, fifth-degree Taylor polynomial on r.
.globl exp_f32
.type exp_f32,@function
exp_f32:
    vmulss xmm1, xmm0, DWORD PTR [rip+.Linvln2]
    vroundss xmm1, xmm1, xmm1, 0
    vcvttss2si eax, xmm1
    vmulss xmm2, xmm1, DWORD PTR [rip+.Lln2hi]
    vsubss xmm0, xmm0, xmm2
    vmulss xmm2, xmm1, DWORD PTR [rip+.Lln2lo]
    vsubss xmm0, xmm0, xmm2
    vmovss xmm2, DWORD PTR [rip+.Lc5]
    vmulss xmm2, xmm2, xmm0
    vaddss xmm2, xmm2, DWORD PTR [rip+.Lc4]
    vmulss xmm2, xmm2, xmm0
    vaddss xmm2, xmm2, DWORD PTR [rip+.Lc3]
    vmulss xmm2, xmm2, xmm0
    vaddss xmm2, xmm2, DWORD PTR [rip+.Lc2]
    vmulss xmm2, xmm2, xmm0
    vaddss xmm2, xmm2, DWORD PTR [rip+.Lone]
    vmulss xmm2, xmm2, xmm0
    vaddss xmm2, xmm2, DWORD PTR [rip+.Lone]
    add eax, 127
    shl eax, 23
    vmovd xmm3, eax
    vmulss xmm0, xmm2, xmm3
    ret
.size exp_f32,.-exp_f32
.section .rodata
.align 4
.align 16
.Lsignmask: .long 0x80000000, 0, 0, 0
.Lplus16: .float 16.0
.Lminus16: .float -16.0
.Lplus80: .float 80.0
.Lminus80: .float -80.0
.Linvln2: .float 1.4426950408889634
.Lln2hi: .float 0.693145751953125
.Lln2lo: .float 0.0000014286068203094172
.Lc5: .float 0.008333333333333333
.Lc4: .float 0.041666666666666664
.Lc3: .float 0.16666666666666666
.Lc2: .float 0.5
.Lone: .float 1.0
.section .note.GNU-stack,"",@progbits
