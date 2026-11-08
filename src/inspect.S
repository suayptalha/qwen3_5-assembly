.intel_syntax noprefix
.section .text
.globl _start
_start:
    cmp QWORD PTR [rsp], 2
    jne usage
    mov rdi, [rsp+16]
    xor esi, esi
    mov eax, 2                   # open(path, O_RDONLY)
    syscall
    test rax, rax
    js fail_open
    mov r12, rax
    mov rdi, r12
    xor esi, esi
    mov edx, 2                   # SEEK_END
    mov eax, 8                   # lseek
    syscall
    cmp rax, 8
    jb fail_file
    mov r13, rax                 # file length
    xor edi, edi
    mov rsi, r13
    mov edx, 1                   # PROT_READ
    mov r10d, 2                  # MAP_PRIVATE
    mov r8, r12
    xor r9d, r9d
    mov eax, 9                   # mmap
    syscall
    cmp rax, -4095
    jae fail_file
    mov r14, rax
    mov r15, [r14]               # safetensors header length, little endian
    mov rax, r13
    sub rax, 8
    cmp r15, rax
    ja fail_file
    cmp r15, 0
    je fail_file
    cmp BYTE PTR [r14+8], '{'
    jne fail_file
    lea r12, [r14+8]
write_loop:
    test r15, r15
    jz ok
    mov eax, 1                   # write(stdout, remaining header)
    mov edi, 1
    mov rsi, r12
    mov rdx, r15
    syscall
    test rax, rax
    jle fail_file
    add r12, rax
    sub r15, rax
    jmp write_loop
ok:
    xor edi, edi
    mov eax, 60
    syscall
usage:
    lea rsi, [rip+usage_msg]
    mov edx, usage_end-usage_msg
    jmp fail
fail_open:
    lea rsi, [rip+open_msg]
    mov edx, open_end-open_msg
    jmp fail
fail_file:
    lea rsi, [rip+file_msg]
    mov edx, file_end-file_msg
fail:
    mov edi, 2
    mov eax, 1
    syscall
    mov edi, 1
    mov eax, 60
    syscall
.section .rodata
usage_msg: .ascii "usage: inspect-safetensors FILE\n"
usage_end:
open_msg: .ascii "cannot open file\n"
open_end:
file_msg: .ascii "invalid or unreadable safetensors header\n"
file_end:
.section .note.GNU-stack,"",@progbits
