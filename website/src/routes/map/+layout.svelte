<script lang="ts">
	import { setMapState } from "$lib/state/map.svelte.js";

	let { children } = $props();

	const map = setMapState();
</script>



<div class="w-screen h-screen absolute flex items-center justify-center">
	{#if map.loading && !map.error}
		<div
			class="rounded-lg bg-accent-foreground text-white px-4 py-3 shadow-lg"
		>
			Loading map...
		</div>
	{:else if map.error}
		<div
			class="rounded-lg bg-white/95 px-4 py-3 text-destructive shadow-lg"
			role="alert"
		>
			{map.error.message}
		</div>
	{/if}
</div>

{@render children()}