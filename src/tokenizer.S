.intel_syntax noprefix
.text
# int64_t tokenize(const char *input, size_t bytes, uint32_t *out,
#                  size_t capacity, const void *tables)
# Scratch storage is static; not thread safe. Returns count, or -1 on error.
.globl tokenize
.type tokenize,@function
tokenize:
    push rbp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 8
    cmp rsi, 1048572
    ja .Ltok_fail
    mov [rip+t_input], rdi
    mov [rip+t_len], rsi
    mov [rip+t_output], rdx
    mov [rip+t_capacity], rcx
    mov [rip+t_data], r8
    mov QWORD PTR [rip+t_outlen], 0
    mov QWORD PTR [rip+t_normalized], 0
.Lrestart_scan:
    xor r12d, r12d               # byte index
    xor r13d, r13d               # character count
.Lscan_utf8:
    cmp r12, [rip+t_len]
    jae .Lscan_end
    lea rax, [rip+t_pos]
    mov DWORD PTR [rax+r13*4], r12d
    mov rdi, [rip+t_input]
    movzx ebx, BYTE PTR [rdi+r12]
    cmp ebx, '<'
    je .Lspecial_search
.Lnormal_utf8:
    cmp ebx, 128
    jb .Lascii_char
    mov eax, ebx
    and eax, 0xe0
    cmp eax, 0xc0
    je .Ltwo
    mov eax, ebx
    and eax, 0xf0
    cmp eax, 0xe0
    je .Lthree
    mov eax, ebx
    and eax, 0xf8
    cmp eax, 0xf0
    je .Lfour
    jmp .Ltok_fail
.Ltwo:
    mov ecx, 2
    and ebx, 0x1f
    jmp .Lutf8_rest
.Lthree:
    mov ecx, 3
    and ebx, 0x0f
    jmp .Lutf8_rest
.Lfour:
    mov ecx, 4
    and ebx, 0x07
.Lutf8_rest:
    lea rax, [r12+rcx]
    cmp rax, [rip+t_len]
    ja .Ltok_fail
    mov edx, 1
.Lutf8_cont:
    movzx eax, BYTE PTR [rdi+r12+1]
    and eax, 0xc0
    cmp eax, 0x80
    jne .Ltok_fail
    movzx eax, BYTE PTR [rdi+r12+1]
    and eax, 0x3f
    shl ebx, 6
    or ebx, eax
    inc r12
    inc edx
    cmp edx, ecx
    jb .Lutf8_cont
    inc r12
    cmp ebx, 0x10ffff
    ja .Ltok_fail
    cmp ebx, 0xd800
    jb .Lutf8_minimum
    cmp ebx, 0xdfff
    jbe .Ltok_fail
.Lutf8_minimum:
    cmp ecx, 2
    jne .Lutf8_min3
    cmp ebx, 0x80
    jb .Ltok_fail
    jmp .Lstore_char
.Lutf8_min3:
    cmp ecx, 3
    jne .Lutf8_min4
    cmp ebx, 0x800
    jb .Ltok_fail
    jmp .Lstore_char
.Lutf8_min4:
    cmp ebx, 0x10000
    jb .Ltok_fail
    jmp .Lstore_char
.Lascii_char:
    inc r12
.Lstore_char:
    lea rax, [rip+t_cp]
    mov DWORD PTR [rax+r13*4], ebx
    mov rax, [rip+t_data]
    mov rdx, [rax+32]
    add rax, rdx
    movzx eax, BYTE PTR [rax+rbx]
    lea rdx, [rip+t_cat]
    mov BYTE PTR [rdx+r13], al
    inc r13
    jmp .Lscan_utf8
.Lspecial_search:
    # Added token spellings are IDs 248044..248069 in the lookup blob.
    mov ebp, 248044
.Lspecial_next:
    cmp ebp, 248070
    jae .Lnormal_utf8
    mov rax, [rip+t_data]
    mov rdx, [rax+16]
    add rdx, rax
    mov ecx, DWORD PTR [rdx+rbp*8+4]
    lea rdx, [r12+rcx]
    cmp rdx, [rip+t_len]
    ja .Lspecial_skip
    mov edx, DWORD PTR [rax+16]
    # token index can be above 4 GiB only in an invalid data file.
    add rdx, rax
    mov edx, DWORD PTR [rdx+rbp*8]
    add rdx, [rax+24]
    add rdx, rax
    xor eax, eax
.Lspecial_cmp:
    cmp eax, ecx
    jae .Lspecial_found
    mov rdi, [rip+t_input]
    mov r8b, BYTE PTR [rdi+r12]
    cmp r8b, BYTE PTR [rdx+rax]
    jne .Lspecial_inc
    inc r12
    inc eax
    jmp .Lspecial_cmp
.Lspecial_inc:
    sub r12, rax               # restore matched prefix length
    inc ebp
    jmp .Lspecial_next
.Lspecial_skip:
    inc ebp
    jmp .Lspecial_next
.Lspecial_found:
    lea rax, [rip+t_cp]
    mov DWORD PTR [rax+r13*4], ebp
    lea rax, [rip+t_cat]
    mov BYTE PTR [rax+r13], 6
    inc r13
    jmp .Lscan_utf8
.Lscan_end:
    lea rax, [rip+t_pos]
    mov DWORD PTR [rax+r13*4], r12d
    cmp QWORD PTR [rip+t_normalized], 0
    jne .Lbegin_split
    mov rdi, r13
    call normalize_nfc
    test rax, rax
    js .Ltok_fail
    lea rdx, [rip+n_bytes]
    mov [rip+t_input], rdx
    mov [rip+t_len], rax
    mov QWORD PTR [rip+t_normalized], 1
    jmp .Lrestart_scan
.Lbegin_split:
    xor r12d, r12d               # current character index
.Lsplit:
    cmp r12, r13
    jae .Ltok_done
    lea rdx, [rip+t_cat]
    movzx eax, BYTE PTR [rdx+r12]
    cmp eax, 6
    je .Lemit_special
    mov r14, r12                 # piece start
    # English contractions, matched case-insensitively before general splitting.
    lea rcx, [rip+t_cp]
    cmp DWORD PTR [rcx+r12*4], 39
    jne .Lword_check
    lea rax, [r12+1]
    cmp rax, r13
    jae .Lword_check
    mov ebx, DWORD PTR [rcx+rax*4]
    or ebx, 32
    cmp ebx, 's'
    je .Lcontract1
    cmp ebx, 't'
    je .Lcontract1
    cmp ebx, 'm'
    je .Lcontract1
    cmp ebx, 'd'
    je .Lcontract1
    lea rax, [r12+2]
    cmp rax, r13
    jae .Lword_check
    mov eax, DWORD PTR [rcx+rax*4]
    or eax, 32
    cmp ebx, 'r'
    jne .Lcontract_ve
    cmp eax, 'e'
    je .Lcontract2
.Lcontract_ve:
    cmp ebx, 'v'
    jne .Lcontract_ll
    cmp eax, 'e'
    je .Lcontract2
.Lcontract_ll:
    cmp ebx, 'l'
    jne .Lword_check
    cmp eax, 'l'
    je .Lcontract2
    jmp .Lword_check
.Lcontract1:
    lea r15, [r12+2]
    jmp .Lpiece
.Lcontract2:
    lea r15, [r12+3]
    jmp .Lpiece
.Lword_check:
    lea rdx, [rip+t_cat]
    movzx eax, BYTE PTR [rdx+r12]
    cmp eax, 1
    je .Lword_start
    cmp eax, 2
    je .Lword_start
    cmp eax, 3
    je .Lnumber
    cmp eax, 5
    je .Lspace_check
    lea rbx, [r12+1]
    cmp rbx, r13
    jae .Lpunct_check
    movzx ecx, BYTE PTR [rdx+rbx]
    cmp ecx, 1
    je .Lword_prefix
    cmp ecx, 2
    je .Lword_prefix
    jmp .Lpunct_check
.Lword_prefix:
    inc r12
.Lword_start:
    mov r15, r12
.Lword_loop:
    cmp r15, r13
    jae .Lpiece
    movzx eax, BYTE PTR [rdx+r15]
    cmp eax, 1
    je .Lword_more
    cmp eax, 2
    jne .Lpiece
.Lword_more:
    inc r15
    jmp .Lword_loop
.Lnumber:
    lea r15, [r12+1]
    jmp .Lpiece
.Lpunct_check:
    lea rcx, [rip+t_cp]
    cmp DWORD PTR [rcx+r12*4], 32
    jne .Lpunct_start
    lea rbx, [r12+1]
    cmp rbx, r13
    jae .Lspace_check
    cmp BYTE PTR [rdx+rbx], 0
    jne .Lspace_check
    inc r12
.Lpunct_start:
    cmp BYTE PTR [rdx+r12], 0
    jne .Lspace_check
    mov r15, r12
.Lpunct_loop:
    cmp r15, r13
    jae .Lpiece
    cmp BYTE PTR [rdx+r15], 0
    jne .Lpunct_newlines
    inc r15
    jmp .Lpunct_loop
.Lpunct_newlines:
    cmp r15, r13
    jae .Lpiece
    cmp BYTE PTR [rdx+r15], 5
    jne .Lpiece
    inc r15
    jmp .Lpunct_newlines
.Lspace_check:
    # Scan a contiguous whitespace run; remember the last CR/LF.
    mov r15, r12
    xor ebx, ebx
.Lspace_loop:
    cmp r15, r13
    jae .Lspace_end
    movzx eax, BYTE PTR [rdx+r15]
    cmp eax, 4
    je .Lspace_more
    cmp eax, 5
    jne .Lspace_end
    lea rbx, [r15+1]
.Lspace_more:
    inc r15
    jmp .Lspace_loop
.Lspace_end:
    test rbx, rbx
    jz .Lspace_no_newline
    mov r15, rbx
    jmp .Lpiece
.Lspace_no_newline:
    cmp r15, r13
    je .Lpiece
    lea rax, [r12+1]
    cmp r15, rax
    jbe .Lpiece
    dec r15
    jmp .Lpiece
.Lemit_special:
    mov rax, [rip+t_outlen]
    cmp rax, [rip+t_capacity]
    jae .Ltok_fail
    lea rdx, [rip+t_cp]
    mov ecx, DWORD PTR [rdx+r12*4]
    mov rdx, [rip+t_output]
    mov DWORD PTR [rdx+rax*4], ecx
    inc QWORD PTR [rip+t_outlen]
    inc r12
    jmp .Lsplit
.Lpiece:
    # r14 = beginning of regex piece, r15 = exclusive end.
    cmp r15, r14
    jbe .Ltok_fail
    mov rdi, r14
    mov rsi, r15
    call tokenize_piece
    test eax, eax
    js .Ltok_fail
    mov r12, r15
    jmp .Lsplit
.Ltok_done:
    mov rax, [rip+t_outlen]
    jmp .Ltok_return
.Ltok_fail:
    mov rax, -1
.Ltok_return:
    add rsp, 8
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

# Convert one regex piece from byte IDs using rank-ordered BPE merges.
# rdi/rsi are start/end character indices.
tokenize_piece:
    push rbp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 8
    lea rax, [rip+t_pos]
    mov r12d, DWORD PTR [rax+rdi*4]
    mov r13d, DWORD PTR [rax+rsi*4]
    xor r14d, r14d
    mov rdx, [rip+t_data]
    mov rax, [rdx]
    add rdx, rax
    mov rdi, [rip+t_input]
    lea rsi, [rip+t_work]
.Lpiece_bytes:
    cmp r12, r13
    jae .Lbpe_pass
    movzx eax, BYTE PTR [rdi+r12]
    mov eax, DWORD PTR [rdx+rax*4]
    mov DWORD PTR [rsi+r14*4], eax
    inc r14
    inc r12
    jmp .Lpiece_bytes
.Lbpe_pass:
    cmp r14, 2
    jb .Lbpe_emit
    xor r12d, r12d
    mov ebx, -1
    xor ebp, ebp
.Lbpe_pairs:
    lea rax, [r12+1]
    cmp rax, r14
    jae .Lbpe_best
    lea rax, [rip+t_work]
    mov edi, DWORD PTR [rax+r12*4]
    mov esi, DWORD PTR [rax+r12*4+4]
    call bpe_lookup
    cmp edx, ebx
    jae .Lbpe_pair_next
    mov ebx, edx
    mov rbp, r12
    mov r15d, eax
.Lbpe_pair_next:
    inc r12
    jmp .Lbpe_pairs
.Lbpe_best:
    cmp ebx, -1
    je .Lbpe_emit
    lea rdx, [rip+t_work]
    mov DWORD PTR [rdx+rbp*4], r15d
    lea rax, [rbp+1]
.Lbpe_shift:
    lea rcx, [rax+1]
    cmp rcx, r14
    jae .Lbpe_shift_done
    mov ecx, DWORD PTR [rdx+rax*4+4]
    mov DWORD PTR [rdx+rax*4], ecx
    inc rax
    jmp .Lbpe_shift
.Lbpe_shift_done:
    dec r14
    jmp .Lbpe_pass
.Lbpe_emit:
    mov rax, [rip+t_outlen]
    lea rdx, [rax+r14]
    cmp rdx, [rip+t_capacity]
    ja .Lbpe_failed
    mov [rip+t_outlen], rdx
    mov rdi, [rip+t_output]
    lea rdi, [rdi+rax*4]
    lea rsi, [rip+t_work]
    mov rcx, r14
    rep movsd
    xor eax, eax
    jmp .Lbpe_return
.Lbpe_failed:
    mov eax, -1
.Lbpe_return:
    add rsp, 8
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

# edi=left ID, esi=right ID; returns eax=result and edx=rank (0xffffffff on miss).
bpe_lookup:
    imul eax, edi, 0x9e3779b1
    xor eax, esi
    mov r8, [rip+t_data]
    and eax, DWORD PTR [r8+40]
    mov r9, [r8+8]
    add r9, r8
.Llookup_probe:
    mov ecx, eax
    shl rcx, 4
    lea r10, [r9+rcx]
    mov edx, DWORD PTR [r10+12]
    cmp edx, -1
    je .Llookup_missing
    cmp DWORD PTR [r10], edi
    jne .Llookup_next
    cmp DWORD PTR [r10+4], esi
    jne .Llookup_next
    mov eax, DWORD PTR [r10+8]
    ret
.Llookup_next:
    inc eax
    and eax, DWORD PTR [r8+40]
    jmp .Llookup_probe
.Llookup_missing:
    xor eax, eax
    ret

# int64_t token_bytes(uint32_t id, const void *tables, const char **ptr)
# returns byte length; -1 for invalid ID.
.globl token_bytes
.type token_bytes,@function
token_bytes:
    cmp rdi, [rsi+48]
    jae .Lbytes_bad
    mov rax, [rsi+16]
    add rax, rsi
    mov ecx, DWORD PTR [rax+rdi*8]
    mov eax, DWORD PTR [rax+rdi*8+4]
    add rcx, [rsi+24]
    add rcx, rsi
    mov [rdx], rcx
    ret
.Lbytes_bad:
    mov rax, -1
    ret
.size token_bytes,.-token_bytes
# NFC normalization of the scanned codepoints. Special tokens are boundaries.
# rdi=number of scanned codepoints; returns normalized UTF-8 byte count or -1.
normalize_nfc:
    push rbp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 8
    mov r15, rdi
    mov rax, [rip+t_data]
    mov rdx, [rax+56]
    add rdx, rax
    mov rcx, [rdx]
    add rcx, rax
    mov [rip+n_decomp], rcx
    mov rcx, [rdx+8]
    add rcx, rax
    mov [rip+n_pool], rcx
    mov rcx, [rdx+16]
    add rcx, rax
    mov [rip+n_ccc], rcx
    mov rcx, [rdx+24]
    add rcx, rax
    mov [rip+n_hash], rcx
    mov rcx, [rdx+32]
    mov [rip+n_mask], rcx
    mov QWORD PTR [rip+n_len], 0
    xor r12d, r12d
.Lnfd_input:
    cmp r12, r15
    jae .Lnfc_begin
    lea rdx, [rip+t_cp]
    mov r14d, DWORD PTR [rdx+r12*4]
    lea rdx, [rip+t_cat]
    cmp BYTE PTR [rdx+r12], 6
    jne .Lnfd_regular
    add r14d, 0x110000
    mov edi, r14d
    call nfc_append
    jmp .Lnfd_next
.Lnfd_regular:
    mov rax, [rip+n_decomp]
    mov ebx, DWORD PTR [rax+r14*4]
    test ebx, ebx
    jnz .Lnfd_expand
    mov edi, r14d
    call nfc_append
    jmp .Lnfd_next
.Lnfd_expand:
    mov ebp, ebx
    and ebp, 31
    shr ebx, 5
    dec ebx
    xor r13d, r13d
.Lnfd_expand_loop:
    cmp r13d, ebp
    jae .Lnfd_next
    mov rax, [rip+n_pool]
    lea edx, [rbx+r13]
    mov edi, DWORD PTR [rax+rdx*4]
    call nfc_append
    inc r13d
    jmp .Lnfd_expand_loop
.Lnfd_next:
    cmp QWORD PTR [rip+n_len], 1048572
    ja .Lnfc_failed
    inc r12
    jmp .Lnfd_input
.Lnfc_begin:
    xor r12d, r12d               # input index
    xor r13d, r13d               # output count
    mov r14, -1                  # most recent starter index
    xor r15d, r15d               # last combining class
.Lnfc_loop:
    cmp r12, [rip+n_len]
    jae .Lnfc_encode
    lea rax, [rip+n_cp]
    mov ebx, DWORD PTR [rax+r12*4]
    xor ebp, ebp
    cmp ebx, 0x110000
    jae .Lnfc_no_compose
    mov rdx, [rip+n_ccc]
    movzx ebp, BYTE PTR [rdx+rbx]
    cmp r14, -1
    je .Lnfc_no_compose
    test r15d, r15d
    jz .Lnfc_try_compose
    cmp r15d, ebp
    jae .Lnfc_no_compose
.Lnfc_try_compose:
    mov edi, DWORD PTR [rax+r14*4]
    mov esi, ebx
    call nfc_compose
    cmp eax, -1
    je .Lnfc_no_compose
    lea rdx, [rip+n_cp]
    mov DWORD PTR [rdx+r14*4], eax
    inc r12
    jmp .Lnfc_loop
.Lnfc_no_compose:
    lea rax, [rip+n_cp]
    mov DWORD PTR [rax+r13*4], ebx
    test ebp, ebp
    jnz .Lnfc_not_starter
    mov r14, r13
    cmp ebx, 0x110000
    jb .Lnfc_not_starter
    mov r14, -1
.Lnfc_not_starter:
    mov r15d, ebp
    inc r13
    inc r12
    jmp .Lnfc_loop
.Lnfc_encode:
    xor r12d, r12d
    xor r14d, r14d               # byte output count
.Lencode_loop:
    cmp r12, r13
    jae .Lencode_done
    lea rax, [rip+n_cp]
    mov ebx, DWORD PTR [rax+r12*4]
    cmp ebx, 0x110000
    jae .Lencode_special
    lea rdi, [rip+n_bytes]
    add rdi, r14
    cmp ebx, 0x80
    jae .Lencode_two_check
    mov BYTE PTR [rdi], bl
    inc r14
    jmp .Lencode_next
.Lencode_two_check:
    cmp ebx, 0x800
    jae .Lencode_three_check
    mov eax, ebx
    shr eax, 6
    or al, 0xc0
    mov BYTE PTR [rdi], al
    mov eax, ebx
    and al, 0x3f
    or al, 0x80
    mov BYTE PTR [rdi+1], al
    add r14, 2
    jmp .Lencode_next
.Lencode_three_check:
    cmp ebx, 0x10000
    jae .Lencode_four
    mov eax, ebx
    shr eax, 12
    or al, 0xe0
    mov BYTE PTR [rdi], al
    mov eax, ebx
    shr eax, 6
    and al, 0x3f
    or al, 0x80
    mov BYTE PTR [rdi+1], al
    mov eax, ebx
    and al, 0x3f
    or al, 0x80
    mov BYTE PTR [rdi+2], al
    add r14, 3
    jmp .Lencode_next
.Lencode_four:
    mov eax, ebx
    shr eax, 18
    or al, 0xf0
    mov BYTE PTR [rdi], al
    mov eax, ebx
    shr eax, 12
    and al, 0x3f
    or al, 0x80
    mov BYTE PTR [rdi+1], al
    mov eax, ebx
    shr eax, 6
    and al, 0x3f
    or al, 0x80
    mov BYTE PTR [rdi+2], al
    mov eax, ebx
    and al, 0x3f
    or al, 0x80
    mov BYTE PTR [rdi+3], al
    add r14, 4
    jmp .Lencode_next
.Lencode_special:
    sub ebx, 0x110000
    mov edi, ebx
    mov rsi, [rip+t_data]
    lea rdx, [rsp]
    call token_bytes@PLT
    test rax, rax
    js .Lnfc_failed
    lea rdx, [r14+rax]
    cmp rdx, 1048572
    ja .Lnfc_failed
    lea rdi, [rip+n_bytes]
    add rdi, r14
    mov rsi, [rsp]
    mov rcx, rax
    rep movsb
    add r14, rax
.Lencode_next:
    cmp r14, 1048572
    ja .Lnfc_failed
    inc r12
    jmp .Lencode_loop
.Lencode_done:
    mov rax, r14
    jmp .Lnfc_return
.Lnfc_failed:
    mov rax, -1
.Lnfc_return:
    add rsp, 8
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
# Append one decomposed codepoint, inserting combining marks in canonical order.
nfc_append:
    mov rax, [rip+n_len]
    cmp rax, 1048572
    jae .Lappend_overflow
    xor ecx, ecx
    cmp edi, 0x110000
    jae .Lappend_insert
    mov rdx, [rip+n_ccc]
    movzx ecx, BYTE PTR [rdx+rdi]
.Lappend_insert:
    lea rdx, [rip+n_cp]
    test ecx, ecx
    jz .Lappend_write
.Lappend_shift:
    test rax, rax
    jz .Lappend_write
    mov r8d, DWORD PTR [rdx+rax*4-4]
    cmp r8d, 0x110000
    jae .Lappend_write
    mov r9, [rip+n_ccc]
    movzx r9d, BYTE PTR [r9+r8]
    test r9d, r9d
    jz .Lappend_write
    cmp r9d, ecx
    jbe .Lappend_write
    mov DWORD PTR [rdx+rax*4], r8d
    dec rax
    jmp .Lappend_shift
.Lappend_write:
    mov DWORD PTR [rdx+rax*4], edi
    inc QWORD PTR [rip+n_len]
    ret
.Lappend_overflow:
    mov QWORD PTR [rip+n_len], 1048573
    ret

# Lookup canonical composition; returns -1 if not composable.
nfc_compose:
    imul eax, edi, 0x9e3779b1
    xor eax, esi
    and eax, DWORD PTR [rip+n_mask]
    mov r8, [rip+n_hash]
.Lcompose_loop:
    lea rdx, [rax+rax*2]
    shl rdx, 2
    lea rcx, [r8+rdx]
    cmp DWORD PTR [rcx], -1
    je .Lcompose_none
    cmp DWORD PTR [rcx], edi
    jne .Lcompose_next
    cmp DWORD PTR [rcx+4], esi
    jne .Lcompose_next
    mov eax, DWORD PTR [rcx+8]
    ret
.Lcompose_next:
    inc eax
    and eax, DWORD PTR [rip+n_mask]
    jmp .Lcompose_loop
.Lcompose_none:
    mov eax, -1
    ret
.bss
.align 16
t_input: .skip 8
t_len: .skip 8
t_output: .skip 8
t_capacity: .skip 8
t_outlen: .skip 8
t_data: .skip 8
t_normalized: .skip 8
n_len: .skip 8
n_decomp: .skip 8
n_pool: .skip 8
n_ccc: .skip 8
n_hash: .skip 8
n_mask: .skip 8
n_cp: .skip 4194304
n_bytes: .skip 1048576
t_pos: .skip 4194304
t_cp: .skip 4194304
t_cat: .skip 1048576
t_work: .skip 4194304
.section .note.GNU-stack,"",@progbits
