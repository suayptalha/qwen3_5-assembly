.intel_syntax noprefix
.text
.globl _start
_start:
.ifdef TEXT_MODE
    mov QWORD PTR [rip+sample_temp], 70
    mov QWORD PTR [rip+sample_k], 40
    mov rax, [rsp]
    cmp rax, 2
    jb usage
    mov rdi, [rsp+16]
    lea rsi, [rip+option_chat]
    call same_string
    test eax, eax
    jnz .Lcli_chat
    mov rdi, [rsp+16]
    lea rsi, [rip+option_sample]
    call same_string
    test eax, eax
    jnz .Lcli_sample
    mov rax, [rsp]
    cmp rax, 5
    je .Ltext_explicit_paths
    cmp rax, 2
    je .Ltext_default_count
    cmp rax, 3
    jne usage
    mov r15, [rsp+24]
    jmp .Ltext_defaults
.Lcli_chat:
    mov QWORD PTR [rip+cli_mode], 1
    mov rax, [rsp]
    cmp rax, 5
    ja usage
    lea r15, [rip+chat_default_count]
    cmp rax, 3
    jb .Lcli_chat_temp
    mov r15, [rsp+24]
.Lcli_chat_temp:
    cmp rax, 4
    jb .Lcli_chat_k
    mov rsi, [rsp+32]
    call parse_num
    jc usage
    cmp BYTE PTR [rsi], 0
    jne usage
    cmp rax, 200
    ja usage
    mov [rip+sample_temp], rax
.Lcli_chat_k:
    cmp QWORD PTR [rsp], 5
    jb .Lcli_chat_open
    mov rsi, [rsp+40]
    call parse_num
    jc usage
    cmp BYTE PTR [rsi], 0
    jne usage
    test rax, rax
    jz usage
    cmp rax, 100
    ja usage
    mov [rip+sample_k], rax
.Lcli_chat_open:
    lea r12, [rip+default_weights]
    lea r13, [rip+default_tables]
    xor r14d, r14d
    jmp .Ltext_open
.Lcli_sample:
    mov QWORD PTR [rip+cli_mode], 2
    mov rax, [rsp]
    cmp rax, 3
    jb usage
    cmp rax, 6
    ja usage
    mov r14, [rsp+24]
    lea r15, [rip+default_count]
    cmp rax, 4
    jb .Lcli_sample_temp
    mov r15, [rsp+32]
.Lcli_sample_temp:
    cmp rax, 5
    jb .Lcli_sample_k
    mov rsi, [rsp+40]
    call parse_num
    jc usage
    cmp BYTE PTR [rsi], 0
    jne usage
    cmp rax, 200
    ja usage
    mov [rip+sample_temp], rax
.Lcli_sample_k:
    cmp QWORD PTR [rsp], 6
    jb .Lcli_sample_open
    mov rsi, [rsp+48]
    call parse_num
    jc usage
    cmp BYTE PTR [rsi], 0
    jne usage
    test rax, rax
    jz usage
    cmp rax, 100
    ja usage
    mov [rip+sample_k], rax
.Lcli_sample_open:
    lea r12, [rip+default_weights]
    lea r13, [rip+default_tables]
    jmp .Ltext_open
.Ltext_default_count:
    lea r15, [rip+default_count]
.Ltext_defaults:
    lea r12, [rip+default_weights]
    lea r13, [rip+default_tables]
    mov r14, [rsp+16]
    jmp .Ltext_open
.Ltext_explicit_paths:
    mov r12, [rsp+16]           # checkpoint
    mov r13, [rsp+24]           # tokenizer tables
    mov r14, [rsp+32]           # UTF-8 prompt
    mov r15, [rsp+40]           # output count
.Ltext_open:
    # Map tokenizer tables read-only.
    mov rdi, r13
    xor esi, esi
    mov eax, 2
    syscall
    test rax, rax
    js failure
    mov r13, rax
    mov rdi, r13
    xor esi, esi
    mov edx, 2
    mov eax, 8
    syscall
    cmp rax, TOKEN_TABLE_BYTES
    jne failure
    mov rsi, rax
    xor edi, edi
    mov edx, 1
    mov r10d, 2
    mov r8, r13
    xor r9d, r9d
    mov eax, 9
    syscall
    cmp rax, -4095
    jae failure
    mov [rip+token_tables], rax
    mov rdi, r13
    mov eax, 3
    syscall
    # Map checkpoint read-only.
    mov rdi, r12
    xor esi, esi
    mov eax, 2
    syscall
    test rax, rax
    js failure
    mov r12, rax
    mov rdi, r12
    xor esi, esi
    mov edx, 2
    mov eax, 8
    syscall
    cmp rax, 1746942600
    jne failure
    mov rsi, rax
    xor edi, edi
    mov edx, 1
    mov r10d, 2
    mov r8, r12
    xor r9d, r9d
    mov eax, 9
    syscall
    cmp rax, -4095
    jae failure
    mov [rip+weights_base], rax
    mov rdi, r12
    mov eax, 3
    syscall
    cmp QWORD PTR [rip+cli_mode], 1
    je .Lchat_initial_count
    mov rdi, r14
    call encode_prompt
    test rax, rax
    js failure
    mov r12, rax
    mov [rip+prompt_len], rax
    mov r14, r15
    jmp .Lparse_common
.Lchat_initial_count:
    mov r14, r15
    xor r12d, r12d
.else
    cmp QWORD PTR [rsp], 4
    jne usage
    mov r12, [rsp+16]
    mov r13, [rsp+24]
    mov r14, [rsp+32]
    mov rdi, r12
    xor esi, esi
    mov eax, 2
    syscall
    test rax, rax
    js failure
    mov r12, rax
    mov rdi, r12
    xor esi, esi
    mov edx, 2
    mov eax, 8
    syscall
    cmp rax, 1746942600
    jne failure
    mov rsi, rax
    xor edi, edi
    mov edx, 1
    mov r10d, 2
    mov r8, r12
    xor r9d, r9d
    mov eax, 9
    syscall
    cmp rax, -4095
    jae failure
    mov [rip+weights_base], rax
    mov rdi, r12
    mov eax, 3
    syscall
    # Parse prompt token IDs, comma-separated.
    mov rsi, r13
    xor r12d, r12d
.Lprompt_parse:
    cmp r12, 262144
    jae failure
    call parse_num
    jc failure
    cmp rax, 248320
    jae failure
    lea rdi, [rip+prompt_tokens]
    mov [rdi+r12*8], rax
    inc r12
    mov al, [rsi]
    test al, al
    jz .Lparsed
    cmp al, ','
    jne failure
    inc rsi
    jmp .Lprompt_parse
.Lparsed:
    mov [rip+prompt_len], r12
.endif
.Lparse_common:
    mov rsi, r14
    call parse_num
    jc failure
    cmp BYTE PTR [rsi], 0
    jne failure
    test rax, rax
    jz failure
    mov [rip+output_count], rax
    add rax, r12
    cmp rax, 262144
    ja failure
    call init_model
    test rax, rax
    js failure
    mov QWORD PTR [rip+position], 0
.ifdef TEXT_MODE
    cmp QWORD PTR [rip+cli_mode], 1
    je .Lchat_read
.endif
.Lbegin_prefill:
    mov QWORD PTR [rip+prefill_index], 0
.Lprefill:
    mov rax, [rip+prefill_index]
    cmp rax, [rip+prompt_len]
    jae .Lgenerate
    call update_pos
    mov rax, [rip+prefill_index]
    lea rdx, [rip+prompt_tokens]
    mov rdi, [rdx+rax*8]
    lea rsi, [rip+model_cfg]
    lea rdx, [rax+1]
    cmp rdx, [rip+prompt_len]
    sete dl
    movzx edx, dl
    call model_step
    mov rdx, [rip+prefill_index]
    inc rdx
    cmp rdx, [rip+prompt_len]
    jne .Lprefill_no_sample
    call pick_token
.Lprefill_no_sample:
    mov [rip+next_token], rax
    inc QWORD PTR [rip+position]
    inc QWORD PTR [rip+prefill_index]
    jmp .Lprefill
.Lgenerate:
    mov QWORD PTR [rip+generated], 0
.Lgenerate_loop:
    mov rdi, [rip+next_token]
    call print_num
    inc QWORD PTR [rip+generated]
    mov rax, [rip+generated]
    cmp rax, [rip+output_count]
    jae .Lturn_done
    mov rax, [rip+next_token]
    cmp rax, 248044
    je .Lturn_done
    cmp rax, 248046
    je .Lturn_done
    call update_pos
    mov rdi, [rip+next_token]
    lea rsi, [rip+model_cfg]
    mov edx, 1
    call model_step
    call pick_token
    mov [rip+next_token], rax
    inc QWORD PTR [rip+position]
    jmp .Lgenerate_loop
.Lturn_done:
.ifdef TEXT_MODE
    cmp QWORD PTR [rip+cli_mode], 1
    je .Lchat_turn_done
.endif
.Lok:
.ifdef TEXT_MODE
    call flush_utf8
    cmp QWORD PTR [rip+cli_mode], 1
    je .Lexit_no_newline
    mov eax, 1
    mov edi, 1
    lea rsi, [rip+newline]
    mov edx, 1
    syscall
.Lexit_no_newline:
.endif
    xor edi, edi
    mov eax, 60
    syscall
usage:
    lea rsi, [rip+usage_text]
    mov edx, usage_end-usage_text
    jmp error
failure:
    lea rsi, [rip+error_text]
    mov edx, error_end-error_text
error:
    mov eax, 1
    mov edi, 2
    syscall
    mov eax, 60
    mov edi, 1
    syscall

# Select next token. Greedy is retained for the default single-prompt mode.
pick_token:
    cmp QWORD PTR [rip+cli_mode], 0
    je .Lpick_done
    cmp QWORD PTR [rip+sample_temp], 0
    je .Lpick_done
    push rbx
    mov eax, 318                 # getrandom
    lea rdi, [rip+random_word]
    mov esi, 4
    xor edx, edx
    syscall
    cmp rax, 4
    je .Lpick_random_ready
    rdtsc                        # fallback when getrandom is unavailable
    mov DWORD PTR [rip+random_word], eax
.Lpick_random_ready:
    lea rdi, [rip+model_logits]
    mov esi, 248320
    mov rdx, [rip+sample_k]
    mov ecx, DWORD PTR [rip+random_word]
    mov rax, [rip+sample_temp]
    vcvtsi2ss xmm0, xmm0, rax
    vdivss xmm0, xmm0, DWORD PTR [rip+temperature_scale]
    call sample_topk
    pop rbx
.Lpick_done:
    ret

.ifdef TEXT_MODE
# Encode one user message with the no-thinking chat template.
# rdi=UTF-8 C string; returns token count or -1.
encode_prompt:
    push r12
    push r13
    sub rsp, 8
    mov r12, rdi
    xor esi, esi
.Lenc_strlen:
    cmp BYTE PTR [r12+rsi], 0
    je .Lenc_tokens
    inc rsi
    cmp rsi, 1048572
    ja .Lenc_bad
    jmp .Lenc_strlen
.Lenc_tokens:
    mov rdi, r12
    lea rdx, [rip+text_ids]
    mov ecx, 262132
    mov r8, [rip+token_tables]
    call tokenize
    test rax, rax
    js .Lenc_bad
    mov r13, rax
    lea rdi, [rip+prompt_tokens]
    mov QWORD PTR [rdi], 248045
    mov QWORD PTR [rdi+8], 846
    mov QWORD PTR [rdi+16], 198
    lea rsi, [rip+text_ids]
    xor ecx, ecx
.Lenc_copy:
    cmp rcx, r13
    jae .Lenc_suffix
    mov eax, DWORD PTR [rsi+rcx*4]
    mov [rdi+rcx*8+24], rax
    inc rcx
    jmp .Lenc_copy
.Lenc_suffix:
    lea rdi, [rdi+rcx*8+24]
    lea rsi, [rip+chat_suffix]
    mov ecx, 9
    rep movsq
    lea rax, [r13+12]
    jmp .Lenc_return
.Lenc_bad:
    mov rax, -1
.Lenc_return:
    add rsp, 8
    pop r13
    pop r12
    ret

.Lchat_read:
    mov eax, 1
    mov edi, 2
    lea rsi, [rip+user_prompt]
    mov edx, user_prompt_end-user_prompt
    syscall
    xor r12d, r12d
.Lchat_read_byte:
    cmp r12, 1048572
    jae failure
    xor eax, eax                 # read stdin, one byte
    xor edi, edi
    lea rsi, [rip+chat_buffer]
    add rsi, r12
    mov edx, 1
    syscall
    test rax, rax
    jle .Lchat_end_input
    lea rdx, [rip+chat_buffer]
    cmp BYTE PTR [rdx+r12], 10
    je .Lchat_line_ready
    inc r12
    jmp .Lchat_read_byte
.Lchat_end_input:
    test r12, r12
    jz .Lok
.Lchat_line_ready:
    lea rdx, [rip+chat_buffer]
    test r12, r12
    jz .Lchat_terminate_line
    cmp BYTE PTR [rdx+r12-1], 13
    jne .Lchat_terminate_line
    dec r12
.Lchat_terminate_line:
    mov BYTE PTR [rdx+r12], 0
    lea rdi, [rip+chat_buffer]
    lea rsi, [rip+exit_command]
    call same_string
    test eax, eax
    jnz .Lok
    lea rdi, [rip+chat_buffer]
    call encode_prompt
    test rax, rax
    js failure
    mov [rip+prompt_len], rax
    add rax, [rip+position]
    add rax, [rip+output_count]
    add rax, 2                   # room for end marker and newline
    cmp rax, 262144
    ja .Lchat_full
    mov eax, 1
    mov edi, 2
    lea rsi, [rip+assistant_prompt]
    mov edx, assistant_prompt_end-assistant_prompt
    syscall
    jmp .Lbegin_prefill
.Lchat_full:
    mov eax, 1
    mov edi, 2
    lea rsi, [rip+context_text]
    mov edx, context_text_end-context_text
    syscall
    jmp .Lok
.Lchat_turn_done:
    call flush_utf8
    mov eax, 1
    mov edi, 1
    lea rsi, [rip+newline]
    mov edx, 1
    syscall
    # The final output token has not yet been fed into the state.
    mov rdi, [rip+next_token]
    cmp rdi, 248044
    je .Lchat_end_marker
    cmp rdi, 248046
    je .Lchat_end_marker
    call chat_feed_token
.Lchat_end_marker:
    mov edi, 248046              # close assistant turn
    call chat_feed_token
    mov edi, 198                 # newline between messages
    call chat_feed_token
    jmp .Lchat_read

chat_feed_token:
    push rbx
    mov rbx, rdi
    call update_pos
    mov rdi, rbx
    lea rsi, [rip+model_cfg]
    xor edx, edx
    call model_step
    inc QWORD PTR [rip+position]
    pop rbx
    ret
.endif

# Return eax=1 for equal NUL-terminated strings, else 0.
same_string:
    xor eax, eax
.Lsame_loop:
    mov dl, BYTE PTR [rdi]
    cmp dl, BYTE PTR [rsi]
    jne .Lsame_done
    inc rdi
    inc rsi
    test dl, dl
    jne .Lsame_loop
    mov eax, 1
.Lsame_done:
    ret

# Parse unsigned decimal from rsi, leave rsi at separator. Carry on failure.
parse_num:
    xor eax, eax
    xor ecx, ecx
.Lparse_loop:
    movzx edx, BYTE PTR [rsi]
    sub edx, '0'
    cmp edx, 9
    ja .Lparse_done
    cmp rax, 100000000
    ja .Lparse_bad
    imul rax, rax, 10
    add rax, rdx
    inc rsi
    inc ecx
    jmp .Lparse_loop
.Lparse_done:
    test ecx, ecx
    jz .Lparse_bad
    clc
    ret
.Lparse_bad:
    stc
    ret

# Print one token ID followed by newline; uses syscall, no libc.
print_num:
.ifdef TEXT_MODE
    cmp rdi, 248044
    jae .Lprint_text_done
    sub rsp, 24
    mov rsi, [rip+token_tables]
    lea rdx, [rsp]
    call token_bytes
    test rax, rax
    js .Lprint_text_error
    mov rsi, rax
    mov rdi, [rsp]
    call emit_utf8
.Lprint_text_error:
    add rsp, 24
.Lprint_text_done:
    ret
.else
    lea rsi, [rip+print_buffer+31]
    mov BYTE PTR [rsi], 10
    mov ecx, 1
    mov rax, rdi
    mov r8d, 10
.Lprint_loop:
    xor edx, edx
    div r8
    add dl, '0'
    dec rsi
    mov [rsi], dl
    inc ecx
    test rax, rax
    jnz .Lprint_loop
    mov edx, ecx
    mov edi, 1
    mov eax, 1
    syscall
    ret
 .endif

# Stream decoded bytes as valid UTF-8, retaining sequences split across tokens.
# Invalid sequences use U+FFFD, matching the usual tokenizer decode behavior.
.ifdef TEXT_MODE
.globl emit_utf8
emit_utf8:
    push rbx
    push r12
    push r13
    push r14
    push r15
    mov r12, rdi
    mov r13, rsi
.Lutf_emit_loop:
    test r13, r13
    jz .Lutf_emit_done
    movzx r14d, BYTE PTR [r12]
.Lutf_reprocess:
    mov ebx, DWORD PTR [rip+utf_n]
    test ebx, ebx
    jnz .Lutf_continuation
    cmp r14d, 0x80
    jb .Lutf_ascii
    cmp r14d, 0xc2
    jb .Lutf_invalid_byte
    cmp r14d, 0xdf
    jbe .Lutf_lead2
    cmp r14d, 0xef
    jbe .Lutf_lead3
    cmp r14d, 0xf4
    jbe .Lutf_lead4
    jmp .Lutf_invalid_byte
.Lutf_lead2:
    mov DWORD PTR [rip+utf_need], 2
    jmp .Lutf_store_lead
.Lutf_lead3:
    mov DWORD PTR [rip+utf_need], 3
    jmp .Lutf_store_lead
.Lutf_lead4:
    mov DWORD PTR [rip+utf_need], 4
.Lutf_store_lead:
    mov BYTE PTR [rip+utf_pending], r14b
    mov DWORD PTR [rip+utf_n], 1
    jmp .Lutf_advance
.Lutf_continuation:
    cmp r14d, 0x80
    jb .Lutf_invalid_pending
    cmp r14d, 0xbf
    ja .Lutf_invalid_pending
    cmp ebx, 1
    jne .Lutf_store_cont
    movzx eax, BYTE PTR [rip+utf_pending]
    cmp eax, 0xe0
    jne .Lutf_check_ed
    cmp r14d, 0xa0
    jb .Lutf_invalid_pending
.Lutf_check_ed:
    cmp eax, 0xed
    jne .Lutf_check_f0
    cmp r14d, 0x9f
    ja .Lutf_invalid_pending
.Lutf_check_f0:
    cmp eax, 0xf0
    jne .Lutf_check_f4
    cmp r14d, 0x90
    jb .Lutf_invalid_pending
.Lutf_check_f4:
    cmp eax, 0xf4
    jne .Lutf_store_cont
    cmp r14d, 0x8f
    ja .Lutf_invalid_pending
.Lutf_store_cont:
    lea rax, [rip+utf_pending]
    mov BYTE PTR [rax+rbx], r14b
    inc ebx
    mov DWORD PTR [rip+utf_n], ebx
    cmp ebx, DWORD PTR [rip+utf_need]
    jne .Lutf_advance
    mov eax, 1
    mov edi, 1
    lea rsi, [rip+utf_pending]
    mov edx, ebx
    syscall
    mov DWORD PTR [rip+utf_n], 0
    jmp .Lutf_advance
.Lutf_ascii:
    mov eax, 1
    mov edi, 1
    mov rsi, r12
    mov edx, 1
    syscall
    jmp .Lutf_advance
.Lutf_invalid_pending:
    mov DWORD PTR [rip+utf_n], 0
    mov eax, 1
    mov edi, 1
    lea rsi, [rip+utf_replacement]
    mov edx, 3
    syscall
    jmp .Lutf_reprocess
.Lutf_invalid_byte:
    mov eax, 1
    mov edi, 1
    lea rsi, [rip+utf_replacement]
    mov edx, 3
    syscall
.Lutf_advance:
    inc r12
    dec r13
    jmp .Lutf_emit_loop
.Lutf_emit_done:
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret
.globl flush_utf8
flush_utf8:
    cmp DWORD PTR [rip+utf_n], 0
    je .Lutf_flush_done
    mov DWORD PTR [rip+utf_n], 0
    mov eax, 1
    mov edi, 1
    lea rsi, [rip+utf_replacement]
    mov edx, 3
    syscall
.Lutf_flush_done:
    ret
.endif

# Recompute sin/cos at current position and update attention cache lengths.
update_pos:
    lea r8, [rip+invfreq]
    lea r9, [rip+cosines]
    lea r10, [rip+sines]
    mov rax, [rip+position]
    vcvtsi2ss xmm0, xmm0, rax
    xor ecx, ecx
    sub rsp, 8
.Langles:
    vmulss xmm1, xmm0, DWORD PTR [r8+rcx*4]
    vmovss DWORD PTR [rsp], xmm1
    fld DWORD PTR [rsp]
    fsincos
    fstp DWORD PTR [r9+rcx*4]
    fstp DWORD PTR [r10+rcx*4]
    inc ecx
    cmp ecx, 32
    jb .Langles
    add rsp, 8
    mov rax, [rip+position]
    inc rax
    xor ecx, ecx
    lea rdx, [rip+layer_states]
.Lcount:
    mov [rdx+48], rax
    add rdx, 64
    inc ecx
    cmp ecx, 24
    jb .Lcount
    ret

# Build pointer tables in BSS; all mutable storage starts zeroed by Linux.
init_model:
    push rbp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 8
    # Reserve 6 GiB of anonymous virtual address space for the six full
    # attention layers. Linux supplies physical pages as they are touched.
    xor edi, edi
    movabs rsi, 6442450944
    mov edx, 3                   # PROT_READ | PROT_WRITE
    mov r10d, 0x22              # MAP_PRIVATE | MAP_ANONYMOUS
    mov r8, -1
    xor r9d, r9d
    mov eax, 9                   # mmap
    syscall
    cmp rax, -4095
    jae .Linit_failed
    mov [rip+kv_base], rax
    lea r12, [rip+model_metadata]
    mov r13, [rip+weights_base]
    mov rax, [r12]
    add rax, r13
    mov [rip+model_cfg], rax
    mov rdi, [r12+8]
    add rdi, r13
    lea rsi, [rip+final_norm]
    mov edx, 1024
    call bf16_to_f32
    lea rax, [rip+final_norm]
    mov [rip+model_cfg+8], rax
    lea rax, [rip+descriptor_ptrs]
    mov [rip+model_cfg+16], rax
    lea rax, [rip+state_ptrs]
    mov [rip+model_cfg+24], rax
    lea rax, [rip+scratch_ptrs]
    mov [rip+model_cfg+32], rax
    lea rax, [rip+mlp_scratch]
    mov [rip+model_cfg+40], rax
    lea rax, [rip+decoder_work]
    mov [rip+model_cfg+48], rax
    lea rax, [rip+model_x]
    mov [rip+model_cfg+56], rax
    lea rax, [rip+model_logits]
    mov [rip+model_cfg+64], rax
    add r12, 16
    xor ebx, ebx
.Linit_layer:
    cmp ebx, 24
    jae .Linit_done
    mov rax, rbx
    imul rax, 56
    lea r14, [rip+descriptors]
    add r14, rax
    lea rdx, [rip+descriptor_ptrs]
    mov [rdx+rbx*8], r14
    mov rax, rbx
    imul rax, 72
    lea r15, [rip+variant_weights]
    add r15, rax
    mov [r14+16], r15
    mov rax, rbx
    imul rax, 12288
    lea rbp, [rip+norm_weights]
    add rbp, rax
    mov [r14], rbp
    lea rax, [rbp+4096]
    mov [r14+8], rax
    mov rdi, [r12]
    add rdi, r13
    mov rsi, rbp
    mov edx, 1024
    call bf16_to_f32
    mov rdi, [r12+8]
    add rdi, r13
    lea rsi, [rbp+4096]
    mov edx, 1024
    call bf16_to_f32
    xor ecx, ecx
.Lvar_ptr:
    mov rax, [r12+rcx*8+16]
    add rax, r13
    mov [r15+rcx*8], rax
    inc ecx
    cmp ecx, 9
    jb .Lvar_ptr
    xor ecx, ecx
.Lmlp_ptr:
    mov rax, [r12+rcx*8+88]
    add rax, r13
    mov [r14+rcx*8+24], rax
    inc ecx
    cmp ecx, 3
    jb .Lmlp_ptr
    mov rax, rbx
    shl rax, 6
    lea rdx, [rip+layer_states]
    add rdx, rax
    lea rcx, [rip+state_ptrs]
    mov [rcx+rbx*8], rdx
    mov rax, rbx
    imul rax, 67200
    lea rcx, [rip+layer_scratch]
    add rcx, rax
    lea rax, [rip+scratch_ptrs]
    mov [rax+rbx*8], rcx
    mov rax, rbx
    and eax, 3
    cmp eax, 3
    je .Linit_attn
    mov QWORD PTR [r14+48], 0
    mov rdi, [r15+48]
    lea rsi, [rbp+10240]
    mov edx, 16
    call bf16_to_f32
    lea rax, [rbp+10240]
    mov [r15+48], rax
    mov rax, rbx
    shl rax, 6
    lea rdx, [rip+layer_states]
    add rdx, rax
    mov rax, rbx
    imul rax, 98304
    lea rcx, [rip+conv_states]
    add rcx, rax
    mov [rdx], rcx
    mov rax, rbx
    shl rax, 20
    lea rcx, [rip+delta_states]
    add rcx, rax
    mov [rdx+8], rcx
    jmp .Linit_next
.Linit_attn:
    mov QWORD PTR [r14+48], 1
    mov rdi, [r15+32]
    lea rsi, [rbp+8192]
    mov edx, 256
    call bf16_to_f32
    lea rax, [rbp+8192]
    mov [r15+32], rax
    mov rdi, [r15+40]
    lea rsi, [rbp+9216]
    mov edx, 256
    call bf16_to_f32
    lea rax, [rbp+9216]
    mov [r15+40], rax
    mov rax, rbx
    shl rax, 6
    lea rdx, [rip+layer_states]
    add rdx, rax
    # Four 256 MiB arrays per attention layer: K0, K1, V0, V1.
    mov rax, rbx
    shr rax, 2
    shl rax, 30
    mov rcx, [rip+kv_base]
    add rcx, rax
    mov [rdx], rcx
    lea rax, [rcx+268435456]
    mov [rdx+8], rax
    lea rax, [rcx+536870912]
    mov [rdx+16], rax
    lea rax, [rcx+805306368]
    mov [rdx+24], rax
    lea rax, [rip+cosines]
    mov [rdx+32], rax
    lea rax, [rip+sines]
    mov [rdx+40], rax
    lea rax, [rip+scores]
    mov [rdx+56], rax
.Linit_next:
    add r12, 112
    inc ebx
    jmp .Linit_layer
.Linit_done:
    xor eax, eax
    jmp .Linit_return
.Linit_failed:
    mov rax, -1
.Linit_return:
    add rsp, 8
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
.section .rodata
 .ifdef TEXT_MODE
usage_text: .ascii "usage: qwen PROMPT [OUTPUT_TOKENS]\n       qwen --chat [OUTPUT_TOKENS [TEMP_PERCENT [TOP_K]]]\n       qwen --sample PROMPT [OUTPUT_TOKENS [TEMP_PERCENT [TOP_K]]]\n       qwen WEIGHTS TOKENIZER_TABLES PROMPT OUTPUT_TOKENS\n"
.else
usage_text: .ascii "usage: qwen-ids WEIGHTS COMMA_SEPARATED_IDS OUTPUT_TOKENS\n"
.endif
usage_end:
error_text: .ascii "invalid arguments or model file\n"
error_end:
utf_replacement: .byte 0xef, 0xbf, 0xbd
temperature_scale: .float 100.0
user_prompt: .ascii "You: "
user_prompt_end:
assistant_prompt: .ascii "Qwen: "
assistant_prompt_end:
context_text: .ascii "\nContext limit reached (262144 tokens); start a new session.\n"
context_text_end:
exit_command: .asciz "/exit"
newline: .byte 10
default_count: .asciz "32"
chat_default_count: .asciz "64"
option_chat: .asciz "--chat"
option_sample: .asciz "--sample"
default_weights: .asciz "model/model.safetensors-00001-of-00001.safetensors"
default_tables: .asciz "model/tokenizer-asm.bin"
.align 8
chat_suffix: .quad 248046,198,248045,74455,198,248068,271,248069,271
.align 4
invfreq:
 .long 0x3f800000
 .long 0x3f1ab32b
 .long 0x3ebaf81a
 .long 0x3e61f836
 .long 0x3e088d77
 .long 0x3da50957
 .long 0x3d47763f
 .long 0x3cf11176
 .long 0x3c91ad39
 .long 0x3c301052
 .long 0x3bd4ca14
 .long 0x3b80967d
 .long 0x3b1b690d
 .long 0x3abbd3ec
 .long 0x3a6301e2
 .long 0x3a092e02
 .long 0x39a5cb5f
 .long 0x394860c1
 .long 0x38f22ce3
 .long 0x3892587f
 .long 0x3830df51
 .long 0x37d5c442
 .long 0x37812dac
 .long 0x371c1fc4
 .long 0x36bcb0c1
 .long 0x36640cc6
 .long 0x3609cf4b
 .long 0x35a68e4c
 .long 0x35494c56
 .long 0x34f3499c
 .long 0x3493048e
 .long 0x3431af44
.bss
.align 64
utf_n: .skip 4
utf_need: .skip 4
utf_pending: .skip 4
token_tables: .skip 8
text_ids: .skip 1048576
weights_base: .skip 8
kv_base: .skip 8
prompt_len: .skip 8
output_count: .skip 8
position: .skip 8
next_token: .skip 8
generated: .skip 8
cli_mode: .skip 8
sample_temp: .skip 8
sample_k: .skip 8
random_word: .skip 4
prefill_index: .skip 8
chat_buffer: .skip 1048576
model_cfg: .skip 72
descriptor_ptrs: .skip 192
state_ptrs: .skip 192
scratch_ptrs: .skip 192
descriptors: .skip 1344
variant_weights: .skip 1728
layer_states: .skip 1536
norm_weights: .skip 294912
layer_scratch: .skip 1612800
conv_states: .skip 2359296
delta_states: .skip 25165824
mlp_scratch: .skip 28672
decoder_work: .skip 8192
model_x: .skip 4096
model_logits: .skip 993280
final_norm: .skip 4096
prompt_tokens: .skip 2097152
cosines: .skip 128
sines: .skip 128
scores: .skip 1048576
print_buffer: .skip 32
.section .note.GNU-stack,"",@progbits
