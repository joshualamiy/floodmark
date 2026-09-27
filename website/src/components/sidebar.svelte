<script lang="ts">
	import type { Snippet } from "svelte";
	import * as Drawer from "$components/ui/drawer";
	import { IsMobile } from "$lib/utils/hooks/is-mobile.svelte";
	import ChevronsUp from "@lucide/svelte/icons/chevrons-up";

	let { children, open = $bindable(false) }: { children?: Snippet; open?: boolean } = $props();

	const isMobile = new IsMobile();
</script>

{#if isMobile.current}
	<Drawer.Root bind:open>
		<Drawer.Trigger
			class="fixed bottom-10 left-1/2 z-30 flex -translate-x-1/2 items-center gap-0.5 rounded-full bg-white px-4 py-2 text-sm font-semibold text-slate-900 shadow-lg ring-1 shadow-slate-950/15 ring-slate-950/10"
		>
			<ChevronsUp class="-ml-1" />
			Cameras
		</Drawer.Trigger>
		<Drawer.Content
			class="flex h-[80dvh] max-h-[80dvh] flex-col overflow-hidden rounded-t-3xl bg-white"
		>
			<Drawer.Title class="sr-only">Cameras</Drawer.Title>
			<div class="h-full min-h-0 flex-1 overflow-hidden">
				{@render children?.()}
			</div>
		</Drawer.Content>
	</Drawer.Root>
{:else}
	<div
		class="fixed inset-y-6 right-6 z-20 flex w-96 max-w-[calc(100vw-3rem)] flex-col overflow-hidden rounded-3xl border border-white/70 bg-white shadow-2xl shadow-slate-950/15 backdrop-blur-md"
	>
		{@render children?.()}
	</div>
{/if}
