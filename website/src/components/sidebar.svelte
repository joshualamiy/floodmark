<script lang="ts">
	import type { Snippet } from "svelte";

	let { children }: { children?: Snippet } = $props();
</script>

<div
	class="fixed inset-x-3 top-20 bottom-3 z-50 flex w-auto max-w-none flex-col overflow-hidden rounded-3xl border border-white/70 bg-white/100 shadow-2xl shadow-slate-950/15 backdrop-blur-md sm:inset-y-6 sm:right-6 sm:left-auto sm:w-96 sm:max-w-[calc(100vw-3rem)]"
>
	{@render children?.()}
</div>
