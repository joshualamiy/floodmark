<script lang="ts">
	import { replaceState } from "$app/navigation";
	import { page } from "$app/state";
	import { onMount } from "svelte";
	import Logo from "$components/logo.svelte";
	import CameraCard from "$components/camera-card.svelte";
	import CameraList from "$components/camera-list.svelte";
	import Map from "$components/Map.svelte";
	import Sidebar from "$components/sidebar.svelte";
	import { useMapState } from "$lib/state/map.svelte";
	import { toast } from "svelte-sonner";

	const map = useMapState();
	let mobileDrawerOpen = $state(false);

	function openMobileDrawer() {
		mobileDrawerOpen = true;
	}

	onMount(() => {
		const notification = page.url.searchParams.get("notification");
		if (!notification) return;

		if (notification === "verified") {
			toast.success("Email verified. Flood alerts are now enabled.");
		} else if (notification === "error") {
			toast.error("This verification link is invalid or expired.");
		}

		const cleanUrl = new URL(page.url);
		cleanUrl.searchParams.delete("notification");
		replaceState(cleanUrl, {});
	});
</script>

<svelte:head>
	<title>Floodmark | Atlanta Flood Detection</title>
	<meta name="description" content="Interactive flood detection map for the Atlanta area." />
</svelte:head>

<main class="h-dvh w-full">
	<Logo />
	<Sidebar bind:open={mobileDrawerOpen}>
		{#if map.activeCamera}
			{@const camera = map.activeCamera}
			<CameraCard {camera} />
		{:else}
			<CameraList onCameraSelect={openMobileDrawer} />
		{/if}
	</Sidebar>
	<Map onCameraSelect={openMobileDrawer} />
</main>
