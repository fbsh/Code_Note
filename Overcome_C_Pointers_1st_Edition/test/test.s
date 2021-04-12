	.file	"test.c"
	.text
	.p2align 4
	.globl	swap_add
	.type	swap_add, @function
swap_add:
.LFB0:
	.cfi_startproc
	endbr64
	movl	(%rdi), %eax
	movl	(%rsi), %edx
	movl	%edx, (%rdi)
	movl	%eax, (%rsi)
	addl	%edx, %eax
	ret
	.cfi_endproc
.LFE0:
	.size	swap_add, .-swap_add
	.p2align 4
	.globl	caller
	.type	caller, @function
caller:
.LFB1:
	.cfi_startproc
	endbr64
	movl	$832093, %eax
	ret
	.cfi_endproc
.LFE1:
	.size	caller, .-caller
	.ident	"GCC: (Ubuntu 9.3.0-17ubuntu1~20.04) 9.3.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	 1f - 0f
	.long	 4f - 1f
	.long	 5
0:
	.string	 "GNU"
1:
	.align 8
	.long	 0xc0000002
	.long	 3f - 2f
2:
	.long	 0x3
3:
	.align 8
4:
