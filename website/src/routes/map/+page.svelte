<script lang="ts">
	import Logo from "$components/logo.svelte";
	import CameraCard from "$components/camera-card.svelte";
	import Map from "$components/map.svelte";
	import Sidebar from "$components/sidebar.svelte";
	import { useMapState } from "$lib/state/map.svelte";

	const map = useMapState();
</script>

<svelte:head>
	<title>Floodmark | Atlanta Flood Detection</title>
	<meta name="description" content="Interactive flood detection map for the Atlanta area." />
</svelte:head>

<main class="h-dvh w-full">
	<Logo />
	<Sidebar>
		{#if map.activeCamera}
			{@const camera = map.activeCamera}
			<CameraCard {camera} />
		{/if}
	</Sidebar>
	<Map />
</main>
