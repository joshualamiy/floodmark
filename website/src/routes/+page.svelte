<script lang="ts">
	import Map from "$components/Map.svelte";
</script>

<svelte:head>
	<title>Floodmark | Atlanta Flood Detection</title>
	<meta name="description" content="Interactive flood detection map for the Atlanta area." />
</svelte:head>

<main class="map-page">
	<Map />
</main>

<style>
	.map-page {
		height: 100dvh;
		width: 100%;
	}
</style>
