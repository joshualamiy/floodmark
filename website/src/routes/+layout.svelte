<script lang="ts">
	import "./layout.css";
	import favicon from "$assets/floodmark-o-favicon-32.png";
	import { QueryClientProvider } from "@tanstack/svelte-query";
	import * as Tooltip from "$components/ui/tooltip";
	import { SvelteQueryDevtools } from "@tanstack/svelte-query-devtools";

	let { data, children } = $props();
</script>

<svelte:head><link rel="icon" href={favicon} /></svelte:head>

<QueryClientProvider client={data.queryClient}>
	<Tooltip.Provider>
		{@render children()}
	</Tooltip.Provider>

	<SvelteQueryDevtools />
</QueryClientProvider>

