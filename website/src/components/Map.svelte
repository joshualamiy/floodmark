<script lang="ts">
	import { onMount } from "svelte";
	import {
		Map as MapLibreMap,
		Marker,
		Popup,
		setWorkerUrl,
		type StyleSpecification,
	} from "maplibre-gl";
	import maplibreWorkerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?url";
	import "maplibre-gl/dist/maplibre-gl.css";
	import cameras from "$lib/assets/cameras_25.json";

	setWorkerUrl(maplibreWorkerUrl);

	let mapContainer: HTMLDivElement;
	let map: MapLibreMap | undefined;
	let markers: Marker[] = [];
	let loading = $state(true);
	let errorMessage = $state("");

	onMount(() => {
		let cancelled = false;

		void fetch(
			"https://raw.githubusercontent.com/openmaptiles/osm-bright-gl-style/master/style.json",
		)
			.then((response) => {
				if (!response.ok) throw new Error(`Unable to load map style: ${response.status}`);
				return response.json();
			})
			.then((style: StyleSpecification) => {
				if (cancelled) return;

				const source = style.sources.openmaptiles;
				if (!source || source.type !== "vector" || !("url" in source)) {
					throw new Error("OSM Bright style has no compatible OpenMapTiles source");
				}

				source.url = "https://tiles.openfreemap.org/planet";
				style.glyphs = "https://tiles.openfreemap.org/fonts/{fontstack}/{range}.pbf";

				map = new MapLibreMap({
					container: mapContainer,
					style,
					maxBounds: [
						[-85.2, 33.1],
						[-83.35, 34.5],
					],
					renderWorldCopies: false,
					attributionControl: {
						customAttribution:
							'<a href="https://openfreemap.org/">OpenFreeMap</a> | ' +
							'<a href="https://openmaptiles.org/">OpenMapTiles</a> | ' +
							'<a href="https://www.openstreetmap.org/copyright">OpenStreetMap contributors</a>',
					},
					center: [-84.388, 33.749],
					zoom: 10,
				});

				map.on("load", () => {
					markers = cameras
						.filter(
							({ Latitude, Longitude }) => Number.isFinite(Latitude) && Number.isFinite(Longitude),
						)
						.map((camera) => {
							const marker = new Marker({ color: "#2563eb" })
								.setLngLat([camera.Longitude, camera.Latitude])
								.setPopup(new Popup({ offset: 24 }).setText(`${camera.Name}\n${camera.Location}`))
								.addTo(map!);

							return marker;
						});

					loading = false;
				});
				map.on("error", (event) => {
					const message = event.error?.message ?? "MapLibre could not render the map";
					console.error("MapLibre error", event.error);
					if (loading) errorMessage = message;
				});
			})
			.catch((error: unknown) => {
				console.error("Map style failed to load", error);
				loading = false;
				errorMessage = error instanceof Error ? error.message : "Unable to load the map";
			});

		return () => {
			cancelled = true;
			markers.forEach((marker) => marker.remove());
			map?.remove();
		};
	});
</script>

<div class="relative h-full min-h-96 w-full">
	<div
		bind:this={mapContainer}
		class="h-full w-full"
		role="application"
		aria-label="Interactive map of Atlanta"
	></div>

	{#if loading && !errorMessage}
		<div class="absolute top-4 left-4 z-[1] rounded-lg bg-white/95 px-4 py-3 shadow-lg">
			Loading map...
		</div>
	{:else if errorMessage}
		<div
			class="absolute top-4 left-4 z-[1] rounded-lg bg-white/95 px-4 py-3 text-red-700 shadow-lg"
			role="alert"
		>
			{errorMessage}
		</div>
	{/if}
</div>
